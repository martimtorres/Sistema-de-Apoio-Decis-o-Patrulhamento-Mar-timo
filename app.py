import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import json
from shapely.geometry import Point, mapping
from shapely.ops import nearest_points

from core import (
    PERFIS, ZONAS_POLIGONOS, ZONA_NOMES, ZONA_FAIXAS_NM, COSTA_PONTOS,
    DADOS_EXEMPLO,
    calcular_distancias, preparar_dataframe, calcular_pontuacao,
    gerar_justificativa, atribuir_zonas_pontuais, agregar_por_zona,
    gerar_incidentes_exemplo_pontual, zona_atual_navio,
)

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
            pontos = pd.read_csv(ficheiro)
            st.sidebar.success(f"{len(pontos)} incidentes carregados do ficheiro.")
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

zona_navio = zona_atual_navio(pos_navio)

# ── Layout principal ──────────────────────────────────────────────────────────
col_mapa, col_info = st.columns([1.3, 1])

# ════════════════════════════════════════════════════════════════════════════
# MAPA — FAIXAS DE DISTÂNCIA À COSTA + HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with col_mapa:
    st.subheader("🗺️ Mapa operacional")
    st.caption(
        f"📍 O navio está atualmente na **Zona {zona_navio} — "
        f"{ZONA_NOMES[zona_navio]}** ({ZONA_FAIXAS_NM[zona_navio]} da costa)."
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

    # ── Marcador do navio ────────────────────────────────────────────────────
    folium.Marker(
        [lat, lon],
        popup=f"Navio<br>({lat:.3f}, {lon:.3f})",
        tooltip="🚢 Posição do navio",
        icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
    ).add_to(mapa)

    # ── Linha para a zona recomendada ─────────────────────────────────────────
    zona_top = int(df.iloc[0]['Zona_Patrulha'])
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

    # ── Clustering dinâmico com Leaflet.MarkerCluster ────────────────────────
    # Injeta CSS + JS do MarkerCluster via macro Folium, e depois cria os
    # marcadores individuais (um por incidente pontual, ou N sintéticos por
    # zona quando os dados são agregados), para que o clustering seja feito
    # pelo browser e responda ao zoom.

    # ── Determinar pontos individuais para clustering ─────────────────────────
    # Modo ponto-a-ponto: usa os pontos reais.
    # Modo agregado: distribui pontos representativos dentro da faixa de zona.
    cluster_pontos = []  # lista de dicts: lat, lon, zona, incidente(0/1)

    if pontos_zonados is not None:
        # Dados ponto-a-ponto: cada linha é um incidente real
        for _, r in pontos_zonados.iterrows():
            cluster_pontos.append({
                'lat':      float(r['Lat']),
                'lon':      float(r['Lon']),
                'zona':     int(r['Zona_Patrulha']),
                'acidente': int(r.get('Acidente', 0)),
                'import':   float(r.get('Importancia', 5)),
            })
    else:
        # Dados agregados: distribui pontos sintéticos dentro de cada faixa
        rng2 = np.random.default_rng(seed=99)
        for _, row in df.iterrows():
            zona = int(row['Zona_Patrulha'])
            poly = ZONAS_POLIGONOS.get(zona)
            if poly is None or poly.is_empty:
                continue
            n_inc  = int(row['Num_Incidentes'])
            n_acid = int(row['Acidentes_Ultimo_Ano'])
            minx_z, miny_z, maxx_z, maxy_z = poly.bounds
            # Gera até 60 pontos por zona (máx visual razoável)
            n_total = min(n_inc, 60)
            gerados = 0
            tentativas = 0
            while gerados < n_total and tentativas < n_total * 40:
                tentativas += 1
                px = rng2.uniform(minx_z, maxx_z)
                py = rng2.uniform(miny_z, maxy_z)
                if poly.contains(Point(px, py)):
                    # Marca proporcionalmente como acidente
                    is_acid = 1 if gerados < n_acid else 0
                    cluster_pontos.append({
                        'lat':      py,
                        'lon':      px,
                        'zona':     zona,
                        'acidente': is_acid,
                        'import':   float(row['Importancia']),
                    })
                    gerados += 1

    # ── Serializa pontos como JSON e injeta JS ────────────────────────────────
    pontos_json = json.dumps(cluster_pontos)

    # Nomes de zonas e faixas para popup
    zona_nomes_json  = json.dumps(ZONA_NOMES)
    zona_faixas_json = json.dumps(ZONA_FAIXAS_NM)

    # Estatísticas por zona para o popup (incidentes totais, acidentes totais, gravidade)
    stats_zona = {}
    for _, row in df.iterrows():
        z = int(row['Zona_Patrulha'])
        stats_zona[z] = {
            'incidentes': int(row['Num_Incidentes']),
            'acidentes':  int(row['Acidentes_Ultimo_Ano']),
            'gravidade':  round(float(row['Importancia']), 1),
            'pontuacao':  round(float(row['Pontuacao']), 3),
        }
    stats_json = json.dumps(stats_zona)

    # Macro que injeta o CSS+JS do MarkerCluster e os clusters numéricos
    cluster_macro = folium.MacroElement()
    cluster_macro._template = folium.utilities.parse_options(
        name='cluster_macro',
        container='body',
    )
    cluster_macro._template = folium.MacroElement()._template

    # Injeção direta via Element
    from folium import Element

    css_cdn = (
        '<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css"/>'
        '<link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css"/>'
    )
    js_cdn = '<script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>'

    mapa.get_root().header.add_child(Element(css_cdn))
    mapa.get_root().header.add_child(Element(js_cdn))

    # CSS personalizado para os clusters numéricos
    custom_css = """
    <style>
    .cluster-incidentes {
        background: radial-gradient(circle, rgba(244,109,67,0.92) 0%, rgba(165,0,38,0.88) 100%);
        border: 3px solid #fff;
        border-radius: 50%;
        color: #fff;
        font-weight: 800;
        font-size: 13px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        line-height: 1.1;
        box-shadow: 0 2px 8px rgba(0,0,0,0.45);
    }
    .cluster-incidentes .cl-inc { font-size: 13px; font-weight: 900; }
    .cluster-incidentes .cl-acid { font-size: 9px; opacity: 0.88; margin-top: 1px; }
    .marker-incidente {
        background: rgba(244,109,67,0.85);
        border: 2px solid #fff;
        border-radius: 50%;
        width: 26px; height: 26px;
        display: flex; align-items: center; justify-content: center;
        font-size: 10px; font-weight: 800; color: #fff;
        box-shadow: 0 1px 5px rgba(0,0,0,0.4);
    }
    .marker-acidente {
        background: rgba(165,0,38,0.9);
        border: 2px solid #ffcc00;
        border-radius: 50%;
        width: 28px; height: 28px;
        display: flex; align-items: center; justify-content: center;
        font-size: 10px; font-weight: 900; color: #fff;
        box-shadow: 0 1px 5px rgba(0,0,0,0.5);
    }
    </style>
    """
    mapa.get_root().header.add_child(Element(custom_css))

    # JS que cria o MarkerClusterGroup e popula com os pontos
    cluster_js = f"""
    <script>
    (function() {{
        // Aguarda o mapa Folium estar disponível
        function initClusters() {{
            // Tenta encontrar o mapa Leaflet criado pelo Folium
            var mapObj = null;
            for (var key in window) {{
                if (window[key] && window[key]._leaflet_id !== undefined &&
                    typeof window[key].addLayer === 'function') {{
                    mapObj = window[key];
                    break;
                }}
            }}
            if (!mapObj) {{ setTimeout(initClusters, 200); return; }}

            var pontos      = {pontos_json};
            var zonaNomes   = {zona_nomes_json};
            var zonaFaixas  = {zona_faixas_json};
            var statsZona   = {stats_json};

            // Configuração do cluster
            var clusterGroup = L.markerClusterGroup({{
                maxClusterRadius: function(zoom) {{
                    if (zoom <= 5)  return 120;
                    if (zoom <= 6)  return 90;
                    if (zoom <= 7)  return 70;
                    if (zoom <= 8)  return 50;
                    if (zoom <= 9)  return 35;
                    if (zoom <= 10) return 20;
                    return 10;
                }},
                iconCreateFunction: function(cluster) {{
                    var markers = cluster.getAllChildMarkers();
                    var totalInc  = markers.length;
                    var totalAcid = markers.filter(function(m) {{ return m.options.isAcidente; }}).length;

                    // Tamanho do círculo conforme quantidade
                    var size = totalInc > 100 ? 58 :
                               totalInc > 50  ? 50 :
                               totalInc > 20  ? 44 :
                               totalInc > 10  ? 38 : 32;

                    var acidLabel = totalAcid > 0
                        ? '<span class="cl-acid">⚠ ' + totalAcid + ' acid.</span>'
                        : '';

                    return L.divIcon({{
                        html: '<div class="cluster-incidentes" style="width:' + size + 'px;height:' + size + 'px;">'
                            + '<span class="cl-inc">' + totalInc + '</span>'
                            + acidLabel
                            + '</div>',
                        className: '',
                        iconSize: [size, size],
                        iconAnchor: [size/2, size/2],
                    }});
                }},
                spiderfyOnMaxZoom: true,
                showCoverageOnHover: false,
                zoomToBoundsOnClick: true,
                animate: true,
                animateAddingMarkers: false,
            }});

            // Adiciona marcador individual por ponto
            pontos.forEach(function(p) {{
                var isAcid = p.acidente === 1;
                var zona   = p.zona;
                var stats  = statsZona[zona] || {{}};
                var nome   = zonaNomes[zona]  || ('Zona ' + zona);
                var faixa  = zonaFaixas[zona] || '';

                var iconHtml = isAcid
                    ? '<div class="marker-acidente">⚠</div>'
                    : '<div class="marker-incidente">●</div>';

                var icon = L.divIcon({{
                    html: iconHtml,
                    className: '',
                    iconSize: [isAcid ? 28 : 26, isAcid ? 28 : 26],
                    iconAnchor: [isAcid ? 14 : 13, isAcid ? 14 : 13],
                }});

                var gravLabel = stats.gravidade !== undefined
                    ? '<tr><td><b>Gravidade média</b></td><td>' + stats.gravidade + '/10</td></tr>'
                    : '';
                var pontLabel = stats.pontuacao !== undefined
                    ? '<tr><td><b>Pontuação</b></td><td>' + stats.pontuacao + '</td></tr>'
                    : '';

                var popupHtml =
                    '<div style="min-width:200px;font-size:13px;">' +
                    '<b style="font-size:14px;">Z' + zona + ' — ' + nome + '</b><br>' +
                    '<span style="color:#777;font-size:11px;">' + faixa + ' da costa</span>' +
                    '<hr style="margin:5px 0;">' +
                    '<table style="width:100%;border-collapse:collapse;">' +
                    '<tr><td><b>Total incidentes</b></td><td><b style="color:#d62728;">' + stats.incidentes + '</b></td></tr>' +
                    '<tr><td><b>Total acidentes</b></td><td><b style="color:#a50026;">' + stats.acidentes + '</b></td></tr>' +
                    gravLabel + pontLabel +
                    '<tr><td colspan="2" style="padding-top:4px;font-size:11px;color:#555;">' +
                    (isAcid ? '⚠️ Este ponto é um <b>acidente</b>' : '📍 Incidente nesta zona') +
                    '</td></tr>' +
                    '</table></div>';

                var marker = L.marker([p.lat, p.lon], {{
                    icon: icon,
                    isAcidente: isAcid,
                }}).bindPopup(popupHtml, {{ maxWidth: 240 }});

                clusterGroup.addLayer(marker);
            }});

            mapObj.addLayer(clusterGroup);
        }}
        setTimeout(initClusters, 400);
    }})();
    </script>
    """
    mapa.get_root().html.add_child(Element(cluster_js))

    st_folium(mapa, height=520, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# RANKING + LEGENDA
# ════════════════════════════════════════════════════════════════════════════
with col_info:
    st.subheader("📋 Ranking")
    zona_top = int(df.iloc[0]['Zona_Patrulha'])
    st.success(
        f"**Recomendação: Zona {zona_top}** — "
        f"{ZONA_NOMES.get(zona_top, '')} ({ZONA_FAIXAS_NM.get(zona_top, '')})  \n"
        f"Distância ao navio: {df.iloc[0]['Distancia']:.0f} km"
    )

    tabela = df[['Zona_Patrulha', 'Distancia', 'Pontuacao']].copy()
    tabela['Faixa'] = tabela['Zona_Patrulha'].map(ZONA_FAIXAS_NM)
    tabela = tabela[['Zona_Patrulha', 'Faixa', 'Distancia', 'Pontuacao']]
    tabela.columns = ['Zona', 'Faixa (à costa)', 'Dist. ao navio (km)', 'Pontuação']
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
