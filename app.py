import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from core import (
    PERFIS, ZONAS_POLIGONOS, DADOS_EXEMPLO,
    calcular_distancias, preparar_dataframe, calcular_pontuacao,
    gerar_justificativa,
)

# ── Nomes descritivos das zonas ──────────────────────────────────────────────
ZONA_NOMES = {
    1: "Águas territoriais (Lisboa)",
    2: "Zona contígua (Algarve)",
    3: "Zona económica exclusiva (Peniche)",
    4: "Plataforma continental (Sotavento)",
    5: "Alto mar costeiro (Figueira da Foz)",
    6: "Atlântico aberto (Açores)",
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
st.sidebar.markdown("**Base de dados**")
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

# ── Cálculos ──────────────────────────────────────────────────────────────────
pos_navio  = (lat, lon)
distancias = calcular_distancias(pos_navio)
df = preparar_dataframe(dados, distancias)
df['Pontuacao'] = calcular_pontuacao(df, pesos)
df = df.sort_values('Pontuacao', ascending=False).reset_index(drop=True)

# ── Layout principal ──────────────────────────────────────────────────────────
col_mapa, col_info = st.columns([1.3, 1])

# ════════════════════════════════════════════════════════════════════════════
# MAPA COM HEATMAP
# ════════════════════════════════════════════════════════════════════════════
with col_mapa:
    st.subheader("🗺️ Mapa operacional")

    # Centra o mapa e ajusta bounds para incluir todas as zonas
    all_coords = []
    for poly in ZONAS_POLIGONOS.values():
        all_coords.extend(poly.exterior.coords)
    all_lats = [c[1] for c in all_coords] + [lat]
    all_lons = [c[0] for c in all_coords] + [lon]
    bounds = [
        [min(all_lats) - 0.3, min(all_lons) - 0.3],
        [max(all_lats) + 0.3, max(all_lons) + 0.3],
    ]

    mapa = folium.Map(location=[lat, lon], tiles="CartoDB positron")
    mapa.fit_bounds(bounds)

    # ── Gerar pontos para o heatmap
    heat_points = []
    rng = np.random.default_rng(seed=42)

    for _, row in df.iterrows():
        zona = int(row['Zona_Patrulha'])
        if zona not in ZONAS_POLIGONOS:
            continue
        poly = ZONAS_POLIGONOS[zona]
        minx, miny, maxx, maxy = poly.bounds

        n_pontos = max(10, int(row['Num_Incidentes'] * row['Importancia'] / 5))
        peso = float(row['Pontuacao'])

        gerados = 0
        tentativas = 0
        while gerados < n_pontos and tentativas < n_pontos * 20:
            tentativas += 1
            px = rng.uniform(minx, maxx)
            py = rng.uniform(miny, maxy)
            from shapely.geometry import Point
            if poly.contains(Point(px, py)):
                heat_points.append([py, px, peso])
                gerados += 1

    HeatMap(
        heat_points,
        min_opacity=0.3,
        max_opacity=0.85,
        radius=28,
        blur=22,
        gradient={
            0.0: "#313695",
            0.3: "#74add1",
            0.5: "#fee090",
            0.7: "#f46d43",
            1.0: "#a50026",
        },
    ).add_to(mapa)

    # ── Marcador do navio
    folium.Marker(
        [lat, lon],
        popup=f"Navio<br>({lat:.3f}, {lon:.3f})",
        tooltip="🚢 Posição do navio",
        icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
    ).add_to(mapa)

    # ── Linha para zona recomendada
    zona_top = int(df.iloc[0]['Zona_Patrulha'])
    if zona_top in ZONAS_POLIGONOS:
        centroide_top = ZONAS_POLIGONOS[zona_top].centroid
        folium.PolyLine(
            [[lat, lon], [centroide_top.y, centroide_top.x]],
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
        f"**Recomendação: Zona {zona_top}** — {ZONA_NOMES.get(zona_top, '')}  \n"
        f"Distância: {df.iloc[0]['Distancia']:.0f} km"
    )

    tabela = df[['Zona_Patrulha', 'Distancia', 'Pontuacao']].copy()
    tabela.columns = ['Zona', 'Dist (km)', 'Pontuação']
    tabela['Dist (km)'] = tabela['Dist (km)'].round(1)
    tabela['Pontuação'] = tabela['Pontuação'].round(3)
    st.dataframe(tabela, hide_index=True, use_container_width=True)

    # ── Legenda das zonas ordenada por Z# ──────────────────────────────
    st.markdown("---")
    st.markdown("**🗂️ Legenda das zonas**")
    for _, row in df.sort_values('Zona_Patrulha').iterrows():
        zona_id  = int(row['Zona_Patrulha'])
        nome     = ZONA_NOMES.get(zona_id, f"Zona {zona_id}")
        acid     = int(row['Acidentes_Ultimo_Ano'])
        inc      = int(row['Num_Incidentes'])
        st.markdown(f"**Z{zona_id} – {nome}:**  \nacidentes {acid} · incidentes {inc}")

# ════════════════════════════════════════════════════════════════════════════
# JUSTIFICATIVAS
# ════════════════════════════════════════════════════════════════════════════
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
