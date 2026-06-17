import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from shapely.geometry import Point, mapping

from core import (
    PERFIS, ZONAS_POLIGONOS, ZONA_NOMES, DADOS_EXEMPLO,
    calcular_distancias, preparar_dataframe, calcular_pontuacao,
    gerar_justificativa, agregar_incidentes_por_zona,
)

st.set_page_config(page_title="Patrulhamento Marítimo", layout="wide")
st.title("🚢 Sistema de Apoio à Decisão — Patrulhamento Marítimo")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Configuração da missão")
perfil = st.sidebar.selectbox("Perfil de operação", list(PERFIS.keys()))
pesos = PERFIS[perfil]

st.sidebar.markdown("**Pesos do perfil**")
for k, v in pesos.items():
    st.sidebar.markdown(f"- {k}: **{v:.0%}**")

st.sidebar.markdown("---")
st.sidebar.markdown("**Posição do navio**")
lat = st.sidebar.number_input("Latitude",  value=38.50, format="%.4f")
lon = st.sidebar.number_input("Longitude", value=-10.00, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("**Base de dados**")
modo = st.sidebar.radio(
    "Tipo de ficheiro",
    ["Agregado por zona", "Incidentes pontuais (lat/lon)"],
)
ficheiro = st.sidebar.file_uploader("Carregar CSV", type=["csv"])

if ficheiro:
    try:
        bruto = pd.read_csv(ficheiro)
        if modo == "Incidentes pontuais (lat/lon)":
            dados = agregar_incidentes_por_zona(bruto)
        else:
            dados = bruto
        st.sidebar.success(f"{len(dados)} zonas carregadas.")
    except Exception as e:
        st.sidebar.error(f"Erro ao ler CSV: {e}")
        dados = DADOS_EXEMPLO
else:
    dados = DADOS_EXEMPLO

# ── Cálculos ─────────────────────────────────────────────────────────────────
pos_navio  = (lat, lon)
distancias = calcular_distancias(pos_navio)
df = preparar_dataframe(dados, distancias)
df['Pontuacao'] = calcular_pontuacao(df, pesos)
df = df.sort_values('Pontuacao', ascending=False).reset_index(drop=True)

# ── Cores das faixas ─────────────────────────────────────────────────────────
CORES_ZONA = {
    1: "#08306b",  # mais costeira → azul escuro
    2: "#2171b5",
    3: "#4292c6",
    4: "#6baed6",
    5: "#9ecae1",
    6: "#deebf7",  # alto mar → azul muito claro
}

col_mapa, col_info = st.columns([1.3, 1])

with col_mapa:
    st.subheader("🗺️ Mapa operacional — faixas marítimas")

    mapa = folium.Map(location=[38.5, -10.5], zoom_start=6,
                      tiles="CartoDB positron")

    # Desenhar cada faixa como GeoJson com tooltip
    for _, row in df.sort_values('Zona_Patrulha').iterrows():
        z = int(row['Zona_Patrulha'])
        geom = ZONAS_POLIGONOS.get(z)
        if geom is None or geom.is_empty:
            continue

        tooltip = (
            f"<b>{ZONA_NOMES[z]}</b><br>"
            f"Incidentes: {int(row['Num_Incidentes'])}<br>"
            f"Acidentes: {int(row['Acidentes_Ultimo_Ano'])}<br>"
            f"Pontuação: {row['Pontuacao']:.3f}"
        )

        folium.GeoJson(
            data=mapping(geom),
            style_function=lambda _, cor=CORES_ZONA[z]: {
                "fillColor": cor,
                "color": "#ffffff",
                "weight": 1,
                "fillOpacity": 0.55,
            },
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(mapa)

    # Heatmap por amostragem dentro de cada anel
    heat_points = []
    rng = np.random.default_rng(seed=42)
    pont_max = df['Pontuacao'].max() or 1.0

    for _, row in df.iterrows():
        z = int(row['Zona_Patrulha'])
        poly = ZONAS_POLIGONOS.get(z)
        if poly is None or poly.is_empty:
            continue

        n_pontos = max(15, int(row['Num_Incidentes'] * row['Importancia'] / 4))
        peso = float(row['Pontuacao']) / pont_max
        minx, miny, maxx, maxy = poly.bounds

        gerados, tentativas = 0, 0
        while gerados < n_pontos and tentativas < n_pontos * 40:
            tentativas += 1
            px = rng.uniform(minx, maxx)
            py = rng.uniform(miny, maxy)
            if poly.contains(Point(px, py)):
                heat_points.append([py, px, peso])
                gerados += 1

    HeatMap(
        heat_points,
        min_opacity=0.25, radius=22, blur=18,
        gradient={
            0.0: "#313695", 0.3: "#74add1", 0.5: "#fee090",
            0.7: "#f46d43", 1.0: "#a50026",
        },
    ).add_to(mapa)

    # Navio
    folium.Marker(
        [lat, lon],
        tooltip=f"🚢 Navio ({lat:.3f}, {lon:.3f})",
        icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
    ).add_to(mapa)

    st_folium(mapa, height=560, use_container_width=True)

# (mantém aqui o resto do teu layout: ranking, legenda, justificativas,
#  decomposição da pontuação — funcionam tal como estavam, porque
#  ZONA_NOMES agora vem de core.py)
