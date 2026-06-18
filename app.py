import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from branca.element import MacroElement, Template
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from pathlib import Path
from shapely.geometry import Point, mapping
from shapely.ops import nearest_points

from core import (
    PERFIS, ZONAS_POLIGONOS, ZONA_NOMES, ZONA_FAIXAS_NM, COSTA_PONTOS,
    DADOS_EXEMPLO, CONSUMO_LITROS_NM_PADRAO,
    calcular_distancias, preparar_dataframe, calcular_pontuacao,
    gerar_justificativa, atribuir_zonas_pontuais, agregar_por_zona,
    gerar_incidentes_exemplo_pontual, zona_atual_navio,
    calcular_autonomia, circulo_autonomia, carregar_incidentes_gama,
    normalizar_incidentes_pontuais,
)

CSV_GAMA_PADRAO = Path(__file__).resolve().parent / "OccurrenceExport-2026-01-19 11_53.csv"
CSV_MARITIMO_PADRAO = Path(__file__).resolve().parent / "OccurrenceExport-maritimo.csv"

# ── Paleta para as faixas de distância (neutra, para não competir com o heatmap)
CORES_ZONA = {
    1: "#dfe7e9",
    2: "#c4d2d6",
    3: "#a8bdc2",
    4: "#8aa3aa",
    5: "#6c8a92",
    6: "#4d6e77",
}

st.set_page_config(page_title="Patrulhamento Marítimo", layout="wide")
st.title("🚢 Sistema de Apoio à Decisão — Patrulhamento Marítimo")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Configuração da missão")

perfil = st.sidebar.selectbox("Perfil de operação", list(PERFIS.keys()))
pesos = PERFIS[perfil]

st.sidebar.markdown("**Pesos do perfil selecionado**")
for k, v in pesos.items():
    st.sidebar.markdown(f"- {k}: **{v:.0%}**")

st.sidebar.markdown("---")
st.sidebar.markdown("**Posição do navio**")
lat = st.sidebar.number_input("Latitude",  value=38.50, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=-9.00, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("**Combustível e autonomia**")
combustivel_litros = st.sidebar.number_input(
    "Combustível disponível (L)",
    min_value=0.0, value=5000.0, step=100.0,
    help="Quantidade de combustível a bordo para calcular o alcance.",
)
consumo_litros_nm = st.sidebar.number_input(
    "Consumo (L/NM)",
    min_value=0.1, value=float(CONSUMO_LITROS_NM_PADRAO), step=0.5,
    help="Litros consumidos por milha náutica percorrida.",
)
autonomia = calcular_autonomia(combustivel_litros, consumo_litros_nm)
st.sidebar.markdown(
    f"- Alcance total: **{autonomia['alcance_total_nm']:.1f} NM** "
    f"({autonomia['alcance_total_km']:.0f} km)\n"
    f"- Raio ida e volta: **{autonomia['raio_ida_volta_nm']:.1f} NM** "
    f"({autonomia['raio_ida_volta_km']:.0f} km)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Base de dados de incidentes**")
formato = st.sidebar.radio(
    "Formato dos dados",
    ["Agregado por zona", "Ponto a ponto (Lat/Lon)"],
    help=(
        "‘Agregado por zona’: um total por zona (comportamento anterior).\n\n"
        "‘Ponto a ponto’: um incidente por linha (Lat/Lon) — a zona é "
        "atribuída automaticamente a partir da distância à costa."
    ),
)

pontos_zonados = None  # só populado no modo ponto a ponto, para mostrar amostra

if formato == "Agregado por zona":
    ficheiro = st.sidebar.file_uploader(
        "Carregar CSV (opcional)", type=["csv"],
        help="Colunas esperadas: Zona_Patrulha, Num_Incidentes, Importancia, Acidentes_Ultimo_Ano"
    )
    if ficheiro:
        try:
            dados = pd.read_csv(ficheiro)
            st.sidebar.success(f"{len(dados)} zonas carregadas do ficheiro.")
        except Exception as e:
            st.sidebar.error(f"Erro ao ler CSV: {e}")
            dados = DADOS_EXEMPLO
    else:
        dados = DADOS_EXEMPLO
else:
    ficheiro = st.sidebar.file_uploader(
        "Carregar CSV ponto a ponto (opcional)", type=["csv"],
        help="Colunas esperadas: Lat, Lon, Importancia (opcional, 1–10), Acidente (opcional, 0/1)"
    )
    try:
        if ficheiro:
            bruto = pd.read_csv(ficheiro)
            n_bruto = len(bruto)
            pontos = normalizar_incidentes_pontuais(bruto)
            st.sidebar.success(
                f"{len(pontos)} incidentes marítimos "
                f"({n_bruto - len(pontos)} em terra/porto removidos)."
            )
        elif CSV_MARITIMO_PADRAO.exists():
            pontos = carregar_incidentes_gama(CSV_MARITIMO_PADRAO)
            st.sidebar.info(f"A usar base marítima limpa ({len(pontos)} incidentes).")
        elif CSV_GAMA_PADRAO.exists():
            pontos = carregar_incidentes_gama(CSV_GAMA_PADRAO)
            st.sidebar.info(
                f"A usar export GAMA filtrado ({len(pontos)} incidentes marítimos)."
            )
        else:
            pontos = gerar_incidentes_exemplo_pontual()
            st.sidebar.info("A usar incidentes de exemplo (ponto a ponto).")
        pontos_zonados = atribuir_zonas_pontuais(pontos)
    except Exception as e:
        st.sidebar.error(f"Erro ao processar CSV ponto a ponto: {e}")
        pontos_zonados = atribuir_zonas_pontuais(gerar_incidentes_exemplo_pontual())

    dados = agregar_por_zona(pontos_zonados)

    with st.sidebar.expander("Ver atribuição automática de zonas (amostra)"):
        amostra = pontos_zonados[['Lat', 'Lon', 'Distancia_Costa_NM', 'Zona_Patrulha']].head(15).copy()
        amostra['Distancia_Costa_NM'] = amostra['Distancia_Costa_NM'].round(1)
        st.dataframe(amostra, hide_index=True, use_container_width=True)

# ── Cálculos ──────────────────────────────────────────────────────────────────
pos_navio  = (lat, lon)
distancias = calcular_distancias(pos_navio)
df = preparar_dataframe(dados, distancias)
df['Pontuacao'] = calcular_pontuacao(df, pesos)
df = df.sort_values('Pontuacao', ascending=False).reset_index(drop=True)

# A distância usada aqui é a distância mínima do navio até à zona.
# Só é considerada operacional se couber no raio de ida-e-volta calculado
# a partir do combustível disponível.
df['Dentro_Autonomia'] = df['Distancia'] <= autonomia['raio_ida_volta_km']
df_alcancavel = df[df['Dentro_Autonomia']]
if not df_alcancavel.empty:
    linha_recomendada = df_alcancavel.iloc[0]
    recomendacao_dentro_autonomia = True
else:
    linha_recomendada = df.iloc[0]
    recomendacao_dentro_autonomia = False
zona_recomendada = int(linha_recomendada['Zona_Patrulha'])

zona_navio = zona_atual_navio(pos_navio)

# ── Layout principal ──────────────────────────────────────────────────────────
col_mapa, col_info = st.columns([1.3, 1])

# ════════════════════════════════════════════════════════════════════════════
# MAPA — FAIXAS DE DISTÂNCIA À COSTA + HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with col_mapa:
    st.subheader("🗺️ Mapa operacional")
    st.caption(
        f"📍 O navio está na **Zona {zona_navio} — "
        f"{ZONA_NOMES[zona_navio]}** ({ZONA_FAIXAS_NM[zona_navio]} da costa).  \n"
        f"⛽ Combustível: **{combustivel_litros:.0f} L** · "
        f"distância total possível **{autonomia['alcance_total_nm']:.1f} NM** · "
        f"raio ida/volta **{autonomia['raio_ida_volta_nm']:.1f} NM** "
        f"({autonomia['raio_ida_volta_km']:.0f} km)."
    )

    # Enquadrar o mapa pelas zonas 1–5 (a zona 6 / alto mar é aberta e
    # tornaria o enquadramento por defeito demasiado afastado).
    zonas_para_bounds = [ZONAS_POLIGONOS[z] for z in range(1, 6) if not ZONAS_POLIGONOS[z].is_empty]
    minx = min(p.bounds[0] for p in zonas_para_bounds + [Point(lon, lat)])
    miny = min(p.bounds[1] for p in zonas_para_bounds + [Point(lon, lat)])
    maxx = max(p.bounds[2] for p in zonas_para_bounds + [Point(lon, lat)])
    maxy = max(p.bounds[3] for p in zonas_para_bounds + [Point(lon, lat)])
    bounds = [[miny - 0.3, minx - 0.3], [maxy + 0.3, maxx + 0.3]]

    mapa = folium.Map(location=[lat, lon], tiles="CartoDB positron")
    mapa.fit_bounds(bounds)

    # ── Gerar pontos para o heatmap dentro de cada faixa ─────────────────────
    # Para cada zona, distribui N pontos aleatórios dentro da faixa marítima
    # correspondente, pesados pelos incidentes e gravidade.
    heat_points = []
    rng = np.random.default_rng(seed=42)

    for _, row in df.iterrows():
        zona = int(row['Zona_Patrulha'])
        poly = ZONAS_POLIGONOS.get(zona)
        if poly is None or poly.is_empty:
            continue
        minx_z, miny_z, maxx_z, maxy_z = poly.bounds

        n_pontos = max(10, int(row['Num_Incidentes'] * row['Importancia'] / 5))
        peso = float(row['Pontuacao'])

        gerados, tentativas = 0, 0
        while gerados < n_pontos and tentativas < n_pontos * 20:
            tentativas += 1
            px = rng.uniform(minx_z, maxx_z)
            py = rng.uniform(miny_z, maxy_z)
            if poly.contains(Point(px, py)):
                heat_points.append([py, px, peso])
                gerados += 1

    HeatMap(
        heat_points,
        name='Heatmap de intensidade',
        min_opacity=0.3,
        max_opacity=0.85,
        radius=24,
        blur=20,
        gradient={
            0.0: "#313695",
            0.3: "#74add1",
            0.5: "#fee090",
            0.7: "#f46d43",
            1.0: "#a50026",
        },
    ).add_to(mapa)

    # ── Faixas de distância (anéis paralelos à costa), com etiqueta ──────────
    for zona in range(1, 7):
        poly = ZONAS_POLIGONOS.get(zona)
        if poly is None or poly.is_empty:
            continue
        linha = df[df['Zona_Patrulha'] == zona]
        incid = int(linha.iloc[0]['Num_Incidentes']) if len(linha) else 0
        acid  = int(linha.iloc[0]['Acidentes_Ultimo_Ano']) if len(linha) else 0
        nome  = ZONA_NOMES[zona]
        faixa = ZONA_FAIXAS_NM[zona]
        cor   = CORES_ZONA[zona]

        label_html = (
            f"<b>Z{zona} — {nome}</b><br>"
            f"{faixa} da costa<br>"
            f"{incid} incidentes · {acid} acidentes"
        )

        folium.GeoJson(
            mapping(poly),
            style_function=lambda _, c=cor: {
                "fillColor": c, "color": c,
                "weight": 1.2, "opacity": 0.55, "fillOpacity": 0.14,
            },
            highlight_function=lambda _: {"weight": 2.2, "fillOpacity": 0.28},
            tooltip=folium.Tooltip(label_html, sticky=True),
        ).add_to(mapa)

    # ── Linha de costa (apenas como limite/referência auxiliar) ──────────────
    folium.PolyLine(
        [[lat_c, lon_c] for lon_c, lat_c in COSTA_PONTOS],
        color="#37474f", weight=1.6, opacity=0.6, dash_array="2 6",
        tooltip="Linha de costa (referência)",
    ).add_to(mapa)

    # ── Círculo de autonomia (ida e volta) ───────────────────────────────────
    if autonomia['raio_ida_volta_km'] > 0:
        raio_m = autonomia['raio_ida_volta_km'] * 1000.0
        folium.Circle(
            location=[lat, lon],
            radius=raio_m,
            color="#1565c0",
            weight=2,
            opacity=0.85,
            fill=True,
            fill_color="#42a5f5",
            fill_opacity=0.12,
            tooltip=(
                f"Autonomia ida e volta: {autonomia['raio_ida_volta_nm']:.1f} NM "
                f"({autonomia['raio_ida_volta_km']:.0f} km)"
            ),
        ).add_to(mapa)

        coords_circulo = circulo_autonomia(lat, lon, autonomia['raio_ida_volta_km'])
        if coords_circulo:
            folium.PolyLine(
                [[c[1], c[0]] for c in coords_circulo],
                color="#1565c0",
                weight=1.5,
                opacity=0.55,
                dash_array="4 6",
                tooltip="Limite de autonomia (ida e volta)",
            ).add_to(mapa)

    # ── Marcador do navio ────────────────────────────────────────────────────
    folium.Marker(
        [lat, lon],
        popup=f"Navio<br>({lat:.3f}, {lon:.3f})",
        tooltip="🚢 Posição do navio",
        icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
    ).add_to(mapa)

    # ── Linha para a zona recomendada ─────────────────────────────────────────
    zona_top = zona_recomendada
    poly_top = ZONAS_POLIGONOS.get(zona_top)
    ponto_navio = Point(lon, lat)
    if poly_top is not None and not poly_top.is_empty and not poly_top.contains(ponto_navio):
        _, alvo = nearest_points(poly_top, ponto_navio)
        folium.PolyLine(
            [[lat, lon], [alvo.y, alvo.x]],
            color="#cc0000", weight=2.5, opacity=0.75,
            tooltip=f"Rota → Zona {zona_top}",
            dash_array="6 4",
        ).add_to(mapa)

    # ══════════════════════════════════════════════════════════════════════════
    # CLUSTERING DINÂMICO — usa folium.plugins.MarkerCluster (nativo)
    # Funciona dentro do iframe do st_folium porque o plugin já inclui o
    # leaflet.markercluster no HTML do mapa gerado pelo Folium.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Estatísticas por zona para os popups ─────────────────────────────────
    stats_zona = {}
    for _, row in df.iterrows():
        z = int(row['Zona_Patrulha'])
        stats_zona[z] = {
            'incidentes': int(row['Num_Incidentes']),
            'acidentes':  int(row['Acidentes_Ultimo_Ano']),
            'gravidade':  round(float(row['Importancia']), 1),
            'pontuacao':  round(float(row['Pontuacao']), 3),
        }

    # ── CSS customizado injectado via MacroElement (fica DENTRO do iframe) ───
    custom_css_macro = MacroElement()
    custom_css_macro._template = Template("""
        {% macro header(this, kwargs) %}
        <style>
        .cluster-custom {
            background: radial-gradient(circle, rgba(244,109,67,0.95) 0%, rgba(165,0,38,0.90) 100%) !important;
            border: 3px solid #fff !important;
            border-radius: 50% !important;
            color: #fff !important;
            font-weight: 900 !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            line-height: 1.1 !important;
            box-shadow: 0 3px 10px rgba(0,0,0,0.5) !important;
        }
        .cl-num  { font-size: 14px; font-weight: 900; }
        .cl-acid { font-size: 9px;  opacity: 0.90; margin-top: 1px; }
        .mk-incidente {
            background: rgba(230,90,40,0.88);
            border: 2px solid #fff;
            border-radius: 50%;
            width: 24px; height: 24px;
            display: flex; align-items: center; justify-content: center;
            font-size: 10px; font-weight: 800; color: #fff;
            box-shadow: 0 1px 5px rgba(0,0,0,0.4);
        }
        .mk-acidente {
            background: rgba(140,0,26,0.92);
            border: 3px solid #ffcc00;
            border-radius: 50%;
            width: 26px; height: 26px;
            display: flex; align-items: center; justify-content: center;
            font-size: 11px; font-weight: 900; color: #fff;
            box-shadow: 0 2px 6px rgba(0,0,0,0.5);
        }
        </style>
        {% endmacro %}
    """)
    mapa.get_root().add_child(custom_css_macro)

    # ── JS para substituir ícones do MarkerCluster após render ───────────────
    # O MarkerCluster nativo usa classes CSS leaflet-marker-icon; interceptamos
    # o evento clusteringend para redesenhar os ícones com os nossos valores.
    cluster_js_macro = MacroElement()
    cluster_js_macro._template = Template("""
        {% macro script(this, kwargs) %}
        (function() {
            // Redesenha cada cluster com contagem numérica customizada
            function styleCluster(cluster) {
                var count = cluster.getChildCount();
                var size  = count > 200 ? 62 : count > 100 ? 54 : count > 50 ? 46 : count > 20 ? 40 : count > 10 ? 34 : 28;
                // Conta acidentes nos filhos (marcadores com classe mk-acidente no HTML)
                var nAcid = 0;
                cluster.getAllChildMarkers().forEach(function(m) {
                    if (m.options && m.options.isAcidente) nAcid++;
                });
                var acidHtml = nAcid > 0 ? '<span class="cl-acid">⚠ '+nAcid+' acid.</span>' : '';
                return L.divIcon({
                    html: '<div class="cluster-custom" style="width:'+size+'px;height:'+size+'px;">'
                        + '<span class="cl-num">'+count+'</span>'
                        + acidHtml
                        + '</div>',
                    className: '',
                    iconSize: L.point(size, size),
                    iconAnchor: L.point(size/2, size/2)
                });
            }
            // Aguarda o mapa e injeta iconCreateFunction em todos os MarkerClusterGroup
            function patchClusters() {
                if (typeof L === 'undefined') { setTimeout(patchClusters, 150); return; }
                // Procura todas as layers do tipo MarkerClusterGroup
                document.querySelectorAll('.leaflet-container').forEach(function(el) {
                    var mapId = el._leaflet_id;
                    if (!mapId) return;
                    // Percorre as layers registadas no Leaflet
                    Object.values(L.map ? {} : {}).forEach(function(){});
                });
                // Abordagem alternativa: escuta eventos globais do Leaflet
                if (window._leaflet_map_patched) return;
                window._leaflet_map_patched = true;
                var origAddLayer = L.Map.prototype.addLayer;
                L.Map.prototype.addLayer = function(layer) {
                    if (layer instanceof L.MarkerClusterGroup) {
                        layer.options.iconCreateFunction = styleCluster;
                        layer.refreshClusters && layer.refreshClusters();
                    }
                    return origAddLayer.apply(this, arguments);
                };
            }
            setTimeout(patchClusters, 100);
        })();
        {% endmacro %}
    """)
    mapa.get_root().add_child(cluster_js_macro)

    # ── Construir pontos para o MarkerCluster ─────────────────────────────────
    cluster_pontos = []

    if pontos_zonados is not None:
        for _, r in pontos_zonados.iterrows():
            cluster_pontos.append({
                'lat':      float(r['Lat']),
                'lon':      float(r['Lon']),
                'zona':     int(r['Zona_Patrulha']),
                'acidente': int(r.get('Acidente', 0)),
                'import':   float(r.get('Importancia', 5)),
            })
    else:
        rng2 = np.random.default_rng(seed=99)
        for _, row in df.iterrows():
            zona   = int(row['Zona_Patrulha'])
            poly   = ZONAS_POLIGONOS.get(zona)
            if poly is None or poly.is_empty:
                continue
            n_inc  = int(row['Num_Incidentes'])
            n_acid = int(row['Acidentes_Ultimo_Ano'])
            n_total = min(n_inc, 80)
            minx_z, miny_z, maxx_z, maxy_z = poly.bounds
            gerados, tentativas = 0, 0
            while gerados < n_total and tentativas < n_total * 40:
                tentativas += 1
                px = rng2.uniform(minx_z, maxx_z)
                py = rng2.uniform(miny_z, maxy_z)
                if poly.contains(Point(px, py)):
                    cluster_pontos.append({
                        'lat':      py,
                        'lon':      px,
                        'zona':     zona,
                        'acidente': 1 if gerados < n_acid else 0,
                        'import':   float(row['Importancia']),
                    })
                    gerados += 1

    # ── Criar o MarkerClusterGroup com opções ─────────────────────────────────
    mc = MarkerCluster(
        options={
            'maxClusterRadius': 60,
            'spiderfyOnMaxZoom': True,
            'showCoverageOnHover': False,
            'zoomToBoundsOnClick': True,
            'animate': True,
            'disableClusteringAtZoom': 12,
        },
        name='Incidentes / Acidentes',
    )

    for p in cluster_pontos:
        zona  = p['zona']
        stats = stats_zona.get(zona, {})
        nome  = ZONA_NOMES.get(zona, f"Zona {zona}")
        faixa = ZONA_FAIXAS_NM.get(zona, "")
        is_acid = p['acidente'] == 1

        icon_html = (
            '<div class="mk-acidente">⚠</div>' if is_acid
            else '<div class="mk-incidente">●</div>'
        )
        icon = folium.DivIcon(
            html=icon_html,
            icon_size=(28 if is_acid else 24, 28 if is_acid else 24),
            icon_anchor=(14 if is_acid else 12, 14 if is_acid else 12),
            class_name='',
        )

        popup_html = f"""
        <div style="min-width:200px;font-size:13px;">
          <b style="font-size:14px;">Z{zona} — {nome}</b><br>
          <span style="color:#777;font-size:11px;">{faixa} da costa</span>
          <hr style="margin:5px 0;">
          <table style="width:100%;border-collapse:collapse;font-size:12px;">
            <tr><td>Total incidentes</td>
                <td><b style="color:#d62728;">{stats.get('incidentes','—')}</b></td></tr>
            <tr><td>Total acidentes</td>
                <td><b style="color:#a50026;">{stats.get('acidentes','—')}</b></td></tr>
            <tr><td>Gravidade média</td>
                <td>{stats.get('gravidade','—')}/10</td></tr>
            <tr><td>Pontuação</td>
                <td>{stats.get('pontuacao','—')}</td></tr>
            <tr><td colspan="2" style="padding-top:5px;font-size:11px;color:#555;">
              {'⚠️ <b>Acidente</b> nesta zona' if is_acid else '📍 Incidente nesta zona'}
            </td></tr>
          </table>
        </div>
        """

        folium.Marker(
            location=[p['lat'], p['lon']],
            icon=icon,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{'⚠ Acidente' if is_acid else 'Incidente'} — Z{zona} {nome}",
        ).add_to(mc)

    mc.add_to(mapa)

    st_folium(mapa, height=520, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# RANKING + LEGENDA
# ════════════════════════════════════════════════════════════════════════════
with col_info:
    st.subheader("📋 Ranking")
    zona_top = zona_recomendada
    distancia_recomendada = float(linha_recomendada['Distancia'])

    if recomendacao_dentro_autonomia:
        st.success(
            f"**Recomendação: Zona {zona_top}** — "
            f"{ZONA_NOMES.get(zona_top, '')} ({ZONA_FAIXAS_NM.get(zona_top, '')})  \n"
            f"Distância ao navio: {distancia_recomendada:.0f} km · "
            f"dentro do raio ida/volta."
        )
    else:
        st.warning(
            f"**Nenhuma zona prioritária está dentro da autonomia ida/volta.**  \n"
            f"Zona de maior risco: Zona {zona_top} — "
            f"{ZONA_NOMES.get(zona_top, '')} ({ZONA_FAIXAS_NM.get(zona_top, '')}), "
            f"a {distancia_recomendada:.0f} km do navio."
        )

    tabela = df[['Zona_Patrulha', 'Distancia', 'Pontuacao', 'Dentro_Autonomia']].copy()
    tabela['Faixa'] = tabela['Zona_Patrulha'].map(ZONA_FAIXAS_NM)
    tabela['Autonomia ida/volta'] = tabela['Dentro_Autonomia'].map({True: 'Sim', False: 'Não'})
    tabela = tabela[['Zona_Patrulha', 'Faixa', 'Distancia', 'Autonomia ida/volta', 'Pontuacao']]
    tabela.columns = ['Zona', 'Faixa (à costa)', 'Dist. ao navio (km)', 'Dentro da autonomia', 'Pontuação']
    tabela['Dist. ao navio (km)'] = tabela['Dist. ao navio (km)'].round(1)
    tabela['Pontuação'] = tabela['Pontuação'].round(3)
    st.dataframe(tabela, hide_index=True, use_container_width=True)

    # ── Legenda das zonas ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🗂️ Legenda das zonas (faixas de distância à costa)**")

    st.markdown(
        """
        <div style="
            display:flex; align-items:center; gap:8px;
            margin-bottom:10px; font-size:12px; color:#555;
        ">
            <span>Baixa intensidade</span>
            <div style="
                flex:1; height:12px; border-radius:6px;
                background: linear-gradient(to right,
                    #313695, #74add1, #fee090, #f46d43, #a50026);
            "></div>
            <span>Alta intensidade</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pontuacao_max = df['Pontuacao'].max()
    for _, row in df.iterrows():
        zona_id  = int(row['Zona_Patrulha'])
        nome     = ZONA_NOMES.get(zona_id, f"Zona {zona_id}")
        faixa    = ZONA_FAIXAS_NM.get(zona_id, "")
        pont     = row['Pontuacao']
        acid     = int(row['Acidentes_Ultimo_Ano'])
        inc      = int(row['Num_Incidentes'])
        ratio    = pont / pontuacao_max if pontuacao_max else 0

        r = int(255)
        g = int(255 - 160 * ratio)
        b = int(255 - 160 * ratio)
        bg = f"rgb({r},{g},{b})"

        st.markdown(
            f"""
            <div style="
                background:{bg}; border-radius:6px;
                padding:6px 10px; margin-bottom:5px;
                border-left: 4px solid {'#a50026' if acid > 0 else '#aaa'};
                font-size:13px;
            ">
                <b>Z{zona_id}</b> — {nome} <span style="color:#777;">({faixa})</span><br>
                <span style="color:#555; font-size:12px;">
                    {inc} incidentes · {acid} acidentes · pontuação {pont:.3f}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# JUSTIFICATIVAS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📝 Justificativas das zonas prioritárias")

justs = gerar_justificativa(df, distancias, pesos, top_k=min(3, len(df)))
for j in justs:
    faixa = ZONA_FAIXAS_NM.get(j['zona'], "")
    st.markdown(
        f"**#{j['posicao']} — Zona {j['zona']} ({faixa})** "
        f"(pontuação **{j['pontuacao']:.3f}**){j['alerta']}  \n"
        f"&nbsp;&nbsp;{j['incidentes']} incidentes históricos · "
        f"gravidade {j['gravidade']}/10 · "
        f"{j['acidentes']} acidentes recentes · "
        f"{j['distancia']:.1f} km ao navio  \n"
        f"&nbsp;&nbsp;Critério dominante: **{j['criterio_dominante']}** "
        f"({j['peso_dominante']:.0%} da pontuação)"
    )

# ════════════════════════════════════════════════════════════════════════════
# DECOMPOSIÇÃO DA PONTUAÇÃO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Decomposição da pontuação")

criterios = {
    'incidentes': ('Num_Incidentes_norm',       'Incidentes',  '#1f77b4'),
    'gravidade':  ('Importancia_norm',          'Gravidade',   '#d62728'),
    'acidentes':  ('Acidentes_Ultimo_Ano_norm', 'Acidentes',   '#ff7f0e'),
    'distancia':  ('Distancia_norm',            'Proximidade', '#2ca02c'),
}
work = df.copy()
for k, (col, _, _) in criterios.items():
    work[f'c_{k}'] = pesos[k] * work[col]
work = work.sort_values('Pontuacao').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 4))
labels   = [f"Z{int(z)}" for z in work['Zona_Patrulha']]
esquerda = np.zeros(len(work))
for k, (_, nome, cor) in criterios.items():
    valores = work[f'c_{k}'].values
    ax.barh(labels, valores, left=esquerda, color=cor,
            label=nome, edgecolor='white')
    esquerda += valores
for i, total in enumerate(work['Pontuacao'].values):
    ax.text(total + 0.005, i, f"{total:.3f}", va='center',
            fontsize=9, fontweight='bold')
ax.set_xlabel('Pontuação')
ax.legend(loc='lower right', fontsize=8)
ax.grid(axis='x', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
st.pyplot(fig)
