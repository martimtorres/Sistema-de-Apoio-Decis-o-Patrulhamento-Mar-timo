import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
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
