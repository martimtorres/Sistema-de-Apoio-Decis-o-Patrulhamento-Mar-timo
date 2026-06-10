import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt

from core import (
    PERFIS, ZONAS_POLIGONOS, DADOS_EXEMPLO,
    calcular_distancias, preparar_dataframe, calcular_pontuacao,
    gerar_justificativa,
)

st.set_page_config(page_title="Patrulhamento Marítimo", layout="wide")
st.title("🚢 Sistema de Apoio à Decisão — Patrulhamento Marítimo")

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

pos_navio = (lat, lon)
distancias = calcular_distancias(pos_navio)

df = preparar_dataframe(DADOS_EXEMPLO, distancias)
df['Pontuacao'] = calcular_pontuacao(df, pesos)
df = df.sort_values('Pontuacao', ascending=False).reset_index(drop=True)

col_mapa, col_info = st.columns([1.3, 1])

with col_mapa:
    st.subheader("🗺️ Mapa operacional")
    centro = [lat, lon]
    mapa = folium.Map(location=centro, zoom_start=7, tiles="CartoDB positron")

    folium.Marker(
        centro,
        popup=f"Navio<br>({lat:.3f}, {lon:.3f})",
        icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
    ).add_to(mapa)

    p_max = df['Pontuacao'].max()
    for _, row in df.iterrows():
        zona = int(row['Zona_Patrulha'])
        if zona not in ZONAS_POLIGONOS:
            continue
        poly = ZONAS_POLIGONOS[zona]
        coords = [(lat_, lon_) for lon_, lat_ in poly.exterior.coords]
        intensidade = row['Pontuacao'] / p_max if p_max else 0
        cor = f"#{int(255*intensidade):02x}{int(80*(1-intensidade)):02x}40"
        folium.Polygon(
            coords, color="black",
            weight=2, fill=True, fill_color=cor, fill_opacity=0.5,
            popup=(f"<b>Zona {zona}</b><br>"
                   f"Pontuação: {row['Pontuacao']:.3f}<br>"
                   f"Distância: {row['Distancia']:.1f} km"),
        ).add_to(mapa)

    zona_top = int(df.iloc[0]['Zona_Patrulha'])
    if zona_top in ZONAS_POLIGONOS:
        centroide_top = ZONAS_POLIGONOS[zona_top].centroid
        folium.PolyLine(
            [centro, (centroide_top.y, centroide_top.x)],
            color="red", weight=3, opacity=0.7,
        ).add_to(mapa)

    st_folium(mapa, height=520, use_container_width=True)

with col_info:
    st.subheader("📋 Ranking")
    zona_top = int(df.iloc[0]['Zona_Patrulha'])
    st.success(f"**Recomendação: Zona {zona_top}** "
               f"(distância {df.iloc[0]['Distancia']:.0f} km)")

    tabela = df[['Zona_Patrulha', 'Distancia', 'Pontuacao']].copy()
    tabela.columns = ['Zona', 'Dist (km)', 'Pontuação']
    tabela['Dist (km)'] = tabela['Dist (km)'].round(1)
    tabela['Pontuação'] = tabela['Pontuação'].round(3)
    st.dataframe(tabela, hide_index=True, use_container_width=True)

st.markdown("---")
st.subheader("📝 Justificativas das zonas prioritárias")

justs = gerar_justificativa(df, distancias, pesos, top_k=min(3, len(df)))
for j in justs:
    st.markdown(
        f"**#{j['posicao']} — Zona {j['zona']}** "
        f"(pontuação **{j['pontuacao']:.3f}**){j['alerta']}  \n"
        f"&nbsp;&nbsp;{j['incidentes']} incidentes históricos · "
        f"gravidade {j['gravidade']}/10 · "
        f"{j['acidentes']} acidentes recentes · "
        f"{j['distancia']:.1f} km  \n"
        f"&nbsp;&nbsp;Critério dominante: **{j['criterio_dominante']}** "
        f"({j['peso_dominante']:.0%} da pontuação)"
    )

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
labels = [f"Zona {int(z)}" for z in work['Zona_Patrulha']]
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