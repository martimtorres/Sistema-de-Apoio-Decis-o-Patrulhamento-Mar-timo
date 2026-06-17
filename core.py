
"""
Sistema de Apoio à Decisão para Patrulhamento Marítimo — núcleo.
Zonas geradas como buffers sucessivos (anéis) a partir da costa portuguesa,
em milhas náuticas, no CRS projetado EPSG:3763 (ETRS89/PT-TM06).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt
from datetime import datetime

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import transform, unary_union
from pyproj import Transformer

# ── Constantes ───────────────────────────────────────────────────────────────
NM_TO_M = 1852.0  # 1 milha náutica em metros

# Limites das zonas em milhas náuticas (cumulativos)
LIMITES_NM = {
    1: (0,   12),
    2: (12,  24),
    3: (24,  50),
    4: (50,  100),
    5: (100, 200),
    6: (200, 400),   # "alto mar" — limitado a 400 NM para o mapa ser desenhável
}

ZONA_NOMES = {
    1: "Águas territoriais (0–12 NM)",
    2: "Zona contígua (12–24 NM)",
    3: "Área costeira alargada (24–50 NM)",
    4: "Zona intermédia da ZEE (50–100 NM)",
    5: "Limite da ZEE (100–200 NM)",
    6: "Alto mar (>200 NM)",
}

# ── Linha de costa simplificada de Portugal continental ──────────────────────
# Pontos (lon, lat) ao longo da costa, de norte para sul.
# Suficiente para gerar buffers realistas; substituível por shapefile real.
COSTA_PT = LineString([
    (-8.87, 41.85),  # Caminha
    (-8.78, 41.69),  # Viana do Castelo
    (-8.82, 41.40),  # Póvoa de Varzim
    (-8.68, 41.18),  # Porto
    (-8.65, 40.64),  # Aveiro
    (-8.86, 40.15),  # Figueira da Foz
    (-9.13, 39.60),  # Peniche
    (-9.42, 39.36),  # Cabo da Roca (aprox.)
    (-9.50, 38.78),  # Cabo Raso
    (-9.21, 38.69),  # Cascais
    (-9.13, 38.70),  # Lisboa
    (-8.89, 38.51),  # Setúbal
    (-8.99, 38.44),  # Sesimbra
    (-8.93, 38.17),  # Sines
    (-8.79, 37.10),  # Sagres
    (-8.40, 37.08),  # Lagos
    (-7.93, 37.01),  # Faro
    (-7.40, 37.18),  # Vila Real de Santo António
])

# Polígono aproximado de Portugal continental (para "subtrair" a terra).
# Simplificado — só precisa de cobrir a massa terrestre.
TERRA_PT = Polygon([
    (-8.87, 41.85), (-8.20, 42.00), (-6.50, 41.90),
    (-6.20, 39.70), (-7.00, 38.20), (-7.40, 37.18),
    (-8.79, 37.10), (-8.99, 38.10), (-9.50, 38.78),
    (-9.42, 39.36), (-9.13, 39.60), (-8.86, 40.15),
    (-8.65, 40.64), (-8.68, 41.18), (-8.82, 41.40),
    (-8.78, 41.69), (-8.87, 41.85),
])

# ── Transformações de CRS ────────────────────────────────────────────────────
# EPSG:4326 (WGS84 lon/lat) ↔ EPSG:3763 (ETRS89 / PT-TM06, metros)
_to_metros = Transformer.from_crs("EPSG:4326", "EPSG:3763", always_xy=True).transform
_to_graus  = Transformer.from_crs("EPSG:3763", "EPSG:4326", always_xy=True).transform


def _projetar(geom):
    return transform(_to_metros, geom)


def _desprojetar(geom):
    return transform(_to_graus, geom)


# ── Construção das zonas como anéis ──────────────────────────────────────────
def _construir_zonas():
    """
    Cria as zonas Z1..Z6 como anéis sucessivos em torno da costa,
    excluindo a terra. Retorna {id_zona: Polygon/MultiPolygon em WGS84}.
    """
    costa_m = _projetar(COSTA_PT)
    terra_m = _projetar(TERRA_PT)

    # Buffers cumulativos (em metros) — discos centrados na linha de costa
    buffers_m = {
        z: costa_m.buffer(lim_ext * NM_TO_M, resolution=64)
        for z, (_, lim_ext) in LIMITES_NM.items()
    }

    zonas = {}
    anterior = None
    for z, (lim_int, _) in LIMITES_NM.items():
        atual = buffers_m[z]
        if lim_int == 0:
            anel = atual
        else:
            buf_int = costa_m.buffer(lim_int * NM_TO_M, resolution=64)
            anel = atual.difference(buf_int)
        # Remover terra
        anel = anel.difference(terra_m)
        zonas[z] = _desprojetar(anel)
        anterior = atual

    return zonas


ZONAS_POLIGONOS = _construir_zonas()


# ── Perfis ───────────────────────────────────────────────────────────────────
PERFIS = {
    'rotina':                          {'incidentes': 0.50, 'gravidade': 0.20, 'acidentes': 0.20, 'distancia': 0.10},
    'emergência':                      {'incidentes': 0.15, 'gravidade': 0.35, 'acidentes': 0.20, 'distancia': 0.30},
    'condições atmosféricas adversas': {'incidentes': 0.25, 'gravidade': 0.25, 'acidentes': 0.10, 'distancia': 0.40},
}


# ── Geometria / distâncias ───────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def distancia_costa_nm(lat, lon):
    """Distância de um ponto (lat, lon) à linha de costa, em milhas náuticas."""
    ponto_m = _projetar(Point(lon, lat))
    costa_m = _projetar(COSTA_PT)
    return ponto_m.distance(costa_m) / NM_TO_M


def zona_por_distancia(dist_nm):
    """Devolve o ID da zona correspondente à distância à costa."""
    for z, (lim_int, lim_ext) in LIMITES_NM.items():
        if lim_int <= dist_nm < lim_ext:
            return z
    return 6  # alto mar por defeito


def distancia_a_zona(pos_navio, zona_id):
    """Distância em km do navio à zona indicada (0 se já está dentro)."""
    lat, lon = pos_navio
    poly = ZONAS_POLIGONOS[zona_id]
    ponto = Point(lon, lat)
    if poly.contains(ponto):
        return 0.0
    # Aproximação: ponto mais próximo na fronteira
    p_proximo = poly.boundary.interpolate(poly.boundary.project(ponto))
    return haversine_km(lat, lon, p_proximo.y, p_proximo.x)


def calcular_distancias(pos_navio, zonas=ZONAS_POLIGONOS):
    return {z: distancia_a_zona(pos_navio, z) for z in zonas}


# ── Normalização e pontuação ─────────────────────────────────────────────────
def normalizar(serie, inverter=False, metodo='linear'):
    s = serie.astype(float).copy()
    if metodo == 'log':
        s = np.log1p(s)
    minimo, maximo = s.min(), s.max()
    if maximo == minimo:
        return pd.Series(0.5, index=s.index)
    norm = (s - minimo) / (maximo - minimo)
    return 1 - norm if inverter else norm


def preparar_dataframe(incidentes, distancias):
    df = incidentes.copy()
    df['Zona_Patrulha'] = df['Zona_Patrulha'].astype(int)
    df['Distancia'] = df['Zona_Patrulha'].map(distancias)
    df['Num_Incidentes_norm']       = normalizar(df['Num_Incidentes'], metodo='log')
    df['Importancia_norm']          = normalizar(df['Importancia'])
    df['Acidentes_Ultimo_Ano_norm'] = normalizar(df['Acidentes_Ultimo_Ano'])
    df['Distancia_norm']            = normalizar(df['Distancia'], inverter=True)
    return df


def calcular_pontuacao(df, pesos):
    return (
        pesos['incidentes'] * df['Num_Incidentes_norm'] +
        pesos['gravidade']  * df['Importancia_norm'] +
        pesos['acidentes']  * df['Acidentes_Ultimo_Ano_norm'] +
        pesos['distancia']  * df['Distancia_norm']
    )


# ── Agregação de incidentes pontuais por distância à costa ───────────────────
def agregar_incidentes_por_zona(incidentes_pontuais: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe DataFrame com colunas: lat, lon, gravidade (1–10), acidente (0/1).
    Calcula a distância à costa de cada ponto, atribui-lhe uma zona e
    agrega contagens por zona.
    """
    if incidentes_pontuais.empty:
        return DADOS_EXEMPLO.copy()

    df = incidentes_pontuais.copy()
    df['dist_nm'] = df.apply(
        lambda r: distancia_costa_nm(r['lat'], r['lon']), axis=1
    )
    df['Zona_Patrulha'] = df['dist_nm'].apply(zona_por_distancia)

    agg = df.groupby('Zona_Patrulha').agg(
        Num_Incidentes=('lat', 'count'),
        Importancia=('gravidade', 'mean'),
        Acidentes_Ultimo_Ano=('acidente', 'sum'),
    ).reset_index()
    # Garantir todas as zonas presentes
    for z in LIMITES_NM:
        if z not in agg['Zona_Patrulha'].values:
            agg = pd.concat([agg, pd.DataFrame([{
                'Zona_Patrulha': z, 'Num_Incidentes': 0,
                'Importancia': 0, 'Acidentes_Ultimo_Ano': 0,
            }])], ignore_index=True)
    return agg.sort_values('Zona_Patrulha').reset_index(drop=True)


# ── Justificativas ───────────────────────────────────────────────────────────
def gerar_justificativa(df, distancias, pesos, top_k=3):
    criterios = {
        'incidentes': ('Num_Incidentes_norm',       'incidentes históricos'),
        'gravidade':  ('Importancia_norm',          'gravidade'),
        'acidentes':  ('Acidentes_Ultimo_Ano_norm', 'acidentes recentes'),
        'distancia':  ('Distancia_norm',            'proximidade'),
    }
    work = df.sort_values('Pontuacao', ascending=False).reset_index(drop=True)
    contribs_cols = {}
    for k, (col, _) in criterios.items():
        contribs_cols[k] = pesos[k] * work[col]

    justificativas = []
    for i in range(min(top_k, len(work))):
        row = work.iloc[i]
        zona = int(row['Zona_Patrulha'])
        contribs = {k: contribs_cols[k].iloc[i] for k in criterios}
        dominante = max(contribs, key=contribs.get)
        peso_dom = contribs[dominante] / row['Pontuacao'] if row['Pontuacao'] else 0
        _, label_dom = criterios[dominante]

        margem = ""
        if i + 1 < len(work):
            diff = row['Pontuacao'] - work.iloc[i + 1]['Pontuacao']
            if row['Pontuacao'] and diff / row['Pontuacao'] < 0.05:
                margem = " ⚠️ decisão apertada"

        justificativas.append({
            'posicao': i + 1,
            'zona': zona,
            'pontuacao': row['Pontuacao'],
            'criterio_dominante': label_dom,
            'peso_dominante': peso_dom,
            'distancia': distancias[zona],
            'incidentes': int(row['Num_Incidentes']),
            'gravidade': row['Importancia'],
            'acidentes': int(row['Acidentes_Ultimo_Ano']),
            'alerta': margem,
        })
    return justificativas


# ── Dados de exemplo agregados (fallback) ────────────────────────────────────
DADOS_EXEMPLO = pd.DataFrame({
    'Zona_Patrulha':        [1, 2, 3, 4, 5, 6],
    'Num_Incidentes':       [120, 85, 200, 40, 60, 30],
    'Importancia':          [8, 6, 9, 4, 5, 3],
    'Acidentes_Ultimo_Ano': [5, 3, 8, 1, 2, 1],
})
