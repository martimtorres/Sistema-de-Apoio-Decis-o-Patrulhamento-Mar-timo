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

    # Ajusta bounds para incluir todas as zonas
    all_coords = []
    for poly in ZONAS_POLIGONOS.values():
        all_coords.extend(poly.exterior.coords)
    all_lats = [c[1] for c in all_coords] + [lat]
    all_lons = [c[0] for c in all_coords] + [lon]
    bounds = [[min(all_lats) - 0.5, min(all_lons) - 0.5],
              [max(all_lats) + 0.5, max(all_lons) + 0.5]]

    mapa = folium.Map(location=centro, tiles="CartoDB positron")
    mapa.fit_bounds(bounds)

    # ------- Marcador do navio -------
    folium.Marker(
        centro,
        popup=f"Navio<br>({lat:.3f}, {lon:.3f})",
        tooltip="🚢 Navio",
        icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
    ).add_to(mapa)

    # ------- Polígonos das zonas coloridos por acidentes -------
    # Garante que TODAS as zonas aparecem, mesmo sem dados
    zonas_com_dados = {int(row['Zona_Patrulha']): row for _, row in df.iterrows()}

    for zona_id, poly in ZONAS_POLIGONOS.items():
        coords = [(lat_, lon_) for lon_, lat_ in poly.exterior.coords]
        centroide = poly.centroid

        if zona_id in zonas_com_dados:
            row = zonas_com_dados[zona_id]
            tem_acidentes = int(row['Acidentes_Ultimo_Ano']) > 0

            if tem_acidentes:
                # Intensidade de vermelho proporcional aos acidentes
                acidentes = int(row['Acidentes_Ultimo_Ano'])
                acidentes_max = int(df['Acidentes_Ultimo_Ano'].max())
                intensidade = acidentes / acidentes_max if acidentes_max else 0
                # Escala do vermelho: rosa claro → vermelho escuro
                r = 180 + int(75 * intensidade)
                g = int(60 * (1 - intensidade))
                b = int(60 * (1 - intensidade))
                fill_color = f"#{r:02x}{g:02x}{b:02x}"
                fill_opacity = 0.45 + 0.3 * intensidade
                border_color = "#8B0000"
            else:
                fill_color = "#cccccc"
                fill_opacity = 0.25
                border_color = "#888888"

            popup_html = (
                f"<b>Zona {zona_id}</b><br>"
                f"Incidentes históricos: {int(row['Num_Incidentes'])}<br>"
                f"Acidentes recentes: {int(row['Acidentes_Ultimo_Ano'])}<br>"
                f"Gravidade: {row['Importancia']}/10<br>"
                f"Pontuação: {row['Pontuacao']:.3f}<br>"
                f"Distância: {row['Distancia']:.1f} km"
            )
            tooltip_html = (
                f"Zona {zona_id} | "
                f"Acidentes: {int(row['Acidentes_Ultimo_Ano'])} | "
                f"Pontuação: {row['Pontuacao']:.3f}"
            )
        else:
            # Zona sem dados na base — cinza transparente
            fill_color = "#cccccc"
            fill_opacity = 0.20
            border_color = "#aaaaaa"
            popup_html = f"<b>Zona {zona_id}</b><br>Sem dados disponíveis"
            tooltip_html = f"Zona {zona_id} | Sem dados"

        folium.Polygon(
            coords,
            color=border_color,
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=fill_opacity,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=tooltip_html,
        ).add_to(mapa)

        # Label com número da zona no centróide
        folium.Marker(
            location=[centroide.y, centroide.x],
            icon=folium.DivIcon(
                html=(
                    f'<div style="'
                    f'font-size:12px;font-weight:bold;color:#222;'
                    f'background:rgba(255,255,255,0.7);'
                    f'border-radius:4px;padding:1px 4px;'
                    f'white-space:nowrap;">'
                    f'Z{zona_id}</div>'
                ),
                icon_size=(30, 18),
                icon_anchor=(15, 9),
            ),
        ).add_to(mapa)

    # ------- Linha para zona top -------
    zona_top = int(df.iloc[0]['Zona_Patrulha'])
    if zona_top in ZONAS_POLIGONOS:
        centroide_top = ZONAS_POLIGONOS[zona_top].centroid
        folium.PolyLine(
            [centro, (centroide_top.y, centroide_top.x)],
            color="red", weight=3, opacity=0.7,
            tooltip="Rota recomendada",
        ).add_to(mapa)

    # ------- Legenda -------
    legenda_html = """
    <div style="
        position: fixed;
        bottom: 30px; right: 30px;
        z-index: 1000;
        background: white;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        line-height: 1.8;
    ">
        <b>Legenda</b><br>
        <span style="display:inline-block;width:16px;height:16px;
            background:#c83c3c;border-radius:3px;vertical-align:middle;
            margin-right:6px;"></span>Zona com acidentes<br>
        <span style="display:inline-block;width:16px;height:16px;
            background:#cccccc;border-radius:3px;vertical-align:middle;
            margin-right:6px;"></span>Zona sem acidentes<br>
        <span style="font-size:11px;color:#555;">
            Intensidade ∝ nº de acidentes
        </span>
    </div>
    """
    mapa.get_root().html.add_child(folium.Element(legenda_html))

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
