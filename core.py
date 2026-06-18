"""
Sistema de Apoio à Decisão para Patrulhamento Marítimo — núcleo.

Inclui: perfis de pesos, decaimento temporal, e divisão do mar em faixas
de distância à costa portuguesa (Z1..Z6), construídas por buffers
sucessivos à linha de costa — não por polígonos desenhados manualmente.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import transform, nearest_points

PERFIS = {
    'rotina':                          {'incidentes': 0.50, 'gravidade': 0.20, 'acidentes': 0.20, 'distancia': 0.10},
    'emergência':                      {'incidentes': 0.15, 'gravidade': 0.35, 'acidentes': 0.20, 'distancia': 0.30},
    'condições atmosféricas adversas': {'incidentes': 0.25, 'gravidade': 0.25, 'acidentes': 0.10, 'distancia': 0.40},
}

# ═══════════════════════════════════════════════════════════════════════════
# LINHA DE COSTA / LINHA DE BASE (aproximação)
# ═══════════════════════════════════════════════════════════════════════════
# Sequência de pontos (lon, lat), norte → sul, que aproxima a linha de costa
# de Portugal continental. É uma simplificação para efeitos de visualização
# e apoio à decisão — para uso operacional/legal, substituir por coordenadas
# oficiais da linha de base (DGRM / IHPT / Decreto-Lei das águas marítimas).
COSTA_PONTOS = [
    (-8.835, 41.873),  # Caminha
    (-8.838, 41.701),  # Viana do Castelo
    (-8.789, 41.536),  # Esposende
    (-8.762, 41.381),  # Póvoa de Varzim
    (-8.711, 41.150),  # Porto / Foz do Douro
    (-8.783, 41.007),  # Espinho
    (-8.737, 40.645),  # Aveiro (Barra)
    (-8.870, 40.151),  # Figueira da Foz
    (-9.069, 39.601),  # Nazaré
    (-9.407, 39.358),  # Peniche / Cabo Carvoeiro
    (-9.498, 38.781),  # Cabo da Roca
    (-9.422, 38.697),  # Cascais
    (-9.217, 38.415),  # Cabo Espichel
    (-8.880, 37.956),  # Sines
    (-8.783, 37.726),  # Vila Nova de Milfontes
    (-8.926, 37.500),  # Zambujeira do Mar
    (-8.972, 37.021),  # Cabo de São Vicente
    (-8.945, 37.006),  # Sagres
    (-8.673, 37.103),  # Lagos
    (-8.250, 37.084),  # Albufeira
    (-7.930, 37.018),  # Faro
    (-7.649, 37.125),  # Tavira
    (-7.418, 37.193),  # Vila Real de Santo António
]
COSTA = LineString(COSTA_PONTOS)

NM_EM_METROS = 1852.0
LIMITES_NM = [12, 24, 50, 100, 200]   # fronteiras Z1|Z2|Z3|Z4|Z5|Z6

ZONA_NOMES = {
    1: "Águas territoriais",
    2: "Zona contígua",
    3: "Área costeira alargada",
    4: "Zona intermédia da ZEE",
    5: "Limite da ZEE",
    6: "Alto mar",
}
ZONA_FAIXAS_NM = {
    1: "0–12 NM",
    2: "12–24 NM",
    3: "24–50 NM",
    4: "50–100 NM",
    5: "100–200 NM",
    6: "> 200 NM",
}

# ── Projeção planar local (equirretangular), só para fazer buffers em metros.
# Erro tipicamente < 1% nesta latitude — aceitável para apoio à decisão,
# não para fins de navegação/cartografia legal.
_LAT_REF = 39.5
_M_POR_GRAU_LAT = 111_320.0
_M_POR_GRAU_LON = 111_320.0 * cos(radians(_LAT_REF))


def _projetar(x, y, z=None):
    return (x * _M_POR_GRAU_LON, y * _M_POR_GRAU_LAT)


def _desprojetar(x, y, z=None):
    return (x / _M_POR_GRAU_LON, y / _M_POR_GRAU_LAT)


def _construir_zonas():
    """Gera as 6 faixas (anéis) de distância à costa por buffers sucessivos."""
    costa_proj = transform(_projetar, COSTA)
    buffers_proj = {d: costa_proj.buffer(d * NM_EM_METROS) for d in LIMITES_NM}
    buffers = {d: transform(_desprojetar, poly) for d, poly in buffers_proj.items()}

    # Máscara grosseira de "terra" (tudo a leste da linha de costa), apenas
    # para impedir que as faixas marítimas invadam visualmente o continente.
    norte, sul = COSTA_PONTOS[0], COSTA_PONTOS[-1]
    massa_terrestre = Polygon(COSTA_PONTOS + [(2.5, sul[1]), (2.5, norte[1])])

    area_interesse = box(-15.0, 35.5, -6.0, 43.0)

    fronteiras = [0] + LIMITES_NM
    zonas = {}
    for i, (inferior, superior) in enumerate(zip(fronteiras[:-1], fronteiras[1:]), start=1):
        anel = buffers[superior] if inferior == 0 else buffers[superior].difference(buffers[inferior])
        zonas[i] = anel.intersection(area_interesse).difference(massa_terrestre)

    zonas[6] = area_interesse.difference(buffers[LIMITES_NM[-1]]).difference(massa_terrestre)
    return zonas


ZONAS_POLIGONOS = _construir_zonas()


# ═══════════════════════════════════════════════════════════════════════════
# DISTÂNCIAS
# ═══════════════════════════════════════════════════════════════════════════
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def distancia_a_poligono(pos_navio, poligono):
    """Distância (km) de um ponto (lat, lon) a um polígono/multipolígono."""
    lat, lon = pos_navio
    ponto = Point(lon, lat)
    if poligono is None or poligono.is_empty:
        return float('nan')
    if poligono.contains(ponto):
        return 0.0
    _, p_proximo = nearest_points(poligono, ponto)
    return haversine_km(lat, lon, p_proximo.y, p_proximo.x)


def calcular_distancias(pos_navio, zonas=ZONAS_POLIGONOS):
    return {z: distancia_a_poligono(pos_navio, poly) for z, poly in zonas.items()}


def distancia_costa_km(lat, lon):
    """Distância (km) de um ponto à linha de costa (não a uma zona)."""
    ponto = Point(lon, lat)
    _, p_proximo = nearest_points(COSTA, ponto)
    return haversine_km(lat, lon, p_proximo.y, p_proximo.x)


def distancia_costa_nm(lat, lon):
    return distancia_costa_km(lat, lon) / 1.852


def zona_por_distancia_nm(distancia_nm):
    """Mapeia uma distância à costa (NM) para a zona 1..6 correspondente."""
    for limite, zona in zip(LIMITES_NM, range(1, len(LIMITES_NM) + 1)):
        if distancia_nm <= limite:
            return zona
    return len(LIMITES_NM) + 1  # zona 6 — alto mar


def zona_atual_navio(pos_navio):
    lat, lon = pos_navio
    return zona_por_distancia_nm(distancia_costa_nm(lat, lon))


# ═══════════════════════════════════════════════════════════════════════════
# INCIDENTES PONTO-A-PONTO (Lat/Lon) → ATRIBUIÇÃO AUTOMÁTICA DE ZONA
# ═══════════════════════════════════════════════════════════════════════════
def atribuir_zonas_pontuais(df_pontos):
    """
    Recebe incidentes individuais com colunas 'Lat' e 'Lon' e calcula, para
    cada um, a distância à costa (NM) e a zona de patrulha (1–6) a que
    pertence, de acordo com as faixas definidas em LIMITES_NM.
    """
    df = df_pontos.copy()
    df['Distancia_Costa_NM'] = [
        distancia_costa_nm(lat, lon) for lat, lon in zip(df['Lat'], df['Lon'])
    ]
    df['Zona_Patrulha'] = df['Distancia_Costa_NM'].apply(zona_por_distancia_nm)
    return df


def agregar_por_zona(df_pontos_com_zona):
    """
    Agrega incidentes ponto-a-ponto (já com 'Zona_Patrulha') para o formato
    usado pelo motor de pontuação: uma linha por zona (1–6), com contagem
    de incidentes, gravidade média e nº de acidentes.
    """
    df = df_pontos_com_zona.copy()
    if 'Importancia' not in df.columns:
        df['Importancia'] = 5.0
    if 'Acidente' not in df.columns:
        df['Acidente'] = 0

    agg = (
        df.groupby('Zona_Patrulha')
        .agg(
            Num_Incidentes=('Zona_Patrulha', 'count'),
            Importancia=('Importancia', 'mean'),
            Acidentes_Ultimo_Ano=('Acidente', 'sum'),
        )
        .reset_index()
    )
    todas_zonas = pd.DataFrame({'Zona_Patrulha': list(ZONA_NOMES.keys())})
    agg = todas_zonas.merge(agg, on='Zona_Patrulha', how='left').fillna(0)
    agg['Zona_Patrulha'] = agg['Zona_Patrulha'].astype(int)
    agg['Num_Incidentes'] = agg['Num_Incidentes'].astype(int)
    agg['Acidentes_Ultimo_Ano'] = agg['Acidentes_Ultimo_Ano'].astype(int)
    return agg


def gerar_incidentes_exemplo_pontual(n_por_zona=None, seed=7):
    """Gera incidentes ponto-a-ponto de exemplo, distribuídos dentro de cada
    faixa de distância à costa — útil para testar o fluxo 'ponto a ponto'."""
    if n_por_zona is None:
        n_por_zona = {1: 40, 2: 25, 3: 18, 4: 10, 5: 6, 6: 3}
    rng = np.random.default_rng(seed)
    registos = []
    for zona, n in n_por_zona.items():
        poly = ZONAS_POLIGONOS.get(zona)
        if poly is None or poly.is_empty:
            continue
        minx, miny, maxx, maxy = poly.bounds
        gerados, tentativas = 0, 0
        while gerados < n and tentativas < n * 50:
            tentativas += 1
            px, py = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            if poly.contains(Point(px, py)):
                registos.append({
                    'Lat': py,
                    'Lon': px,
                    'Importancia': float(rng.integers(1, 11)),
                    'Acidente': int(rng.random() < 0.12),
                })
                gerados += 1
    return pd.DataFrame(registos)


def score_incidentes_com_decaimento(datas_incidentes, lambda_anual=0.3,
                                    hoje: datetime | None = None):
    if hoje is None:
        hoje = datetime.now()
    if len(datas_incidentes) == 0:
        return 0.0
    idades = np.array([(hoje - d).days / 365.25 for d in datas_incidentes])
    return float(np.sum(np.exp(-lambda_anual * idades)))


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


def gerar_justificativa(df, distancias, pesos, top_k=3):
    criterios = {
        'incidentes': ('Num_Incidentes_norm',       'incidentes históricos'),
        'gravidade':  ('Importancia_norm',          'gravidade'),
        'acidentes':  ('Acidentes_Ultimo_Ano_norm', 'acidentes recentes'),
        'distancia':  ('Distancia_norm',            'proximidade'),
    }
    work = df.copy()
    for k, (col, _) in criterios.items():
        work[f'contrib_{k}'] = pesos[k] * work[col]
    work['Pontuacao'] = sum(work[f'contrib_{k}'] for k in criterios)
    work = work.sort_values('Pontuacao', ascending=False).reset_index(drop=True)

    justificativas = []
    for i in range(min(top_k, len(work))):
        row = work.iloc[i]
        zona = int(row['Zona_Patrulha'])
        contribs = {k: row[f'contrib_{k}'] for k in criterios}
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


DADOS_EXEMPLO = pd.DataFrame({
    'Zona_Patrulha':        [1, 2, 3, 4, 5, 6],
    'Num_Incidentes':       [120, 85, 200, 40, 60, 30],
    'Importancia':          [8, 6, 9, 4, 5, 3],
    'Acidentes_Ultimo_Ano': [5, 3, 8, 1, 2, 1],
})
