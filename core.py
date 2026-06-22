"""
Sistema de Apoio à Decisão para Patrulhamento Marítimo — núcleo.

Divisão do mar em faixas de distância à costa portuguesa (Z1..Z6),
construídas por buffers sucessivos à linha de costa.

Secções:
  1. Constantes e configuração
  2. Geometria da costa (Continente, Açores, Madeira)
  3. Construção das zonas de patrulha
  4. Funções de distância e classificação geográfica
  5. Correção de coordenadas costeiras
  6. Leitura e normalização do CSV GAMA
  7. Agregação por zona
  8. Motor de pontuação
  9. Autonomia e navegação
"""

from __future__ import annotations

import re
from datetime import datetime
from math import asin, atan2, cos, degrees, radians, sin, sqrt

import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, transform, unary_union


# ═══════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES E CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

NM_EM_METROS = 1852.0
_M_POR_GRAU_LAT = 111_320.0

# Fronteiras das zonas de patrulha em milhas náuticas
LIMITES_NM = [12, 24, 50, 100, 200]  # Z1 | Z2 | Z3 | Z4 | Z5 | Z6

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

# Perfis de pesos para o motor de pontuação
PERFIS = {
    'rotina':                          {'incidentes': 0.50, 'gravidade': 0.20, 'acidentes': 0.20, 'distancia': 0.10},
    'emergência':                      {'incidentes': 0.20, 'gravidade': 0.35, 'acidentes': 0.30, 'distancia': 0.15},
    'condições atmosféricas adversas': {'incidentes': 0.25, 'gravidade': 0.40, 'acidentes': 0.20, 'distancia': 0.15},
}

# Mapeamento de severidade GAMA → valor numérico de importância
SEVERIDADE_IMPORTANCIA = {
    'Very serious':   10.0,
    'Serious':         8.0,
    'Less Serious':    5.0,
    'Marine incident': 4.0,
}

# Consumo por defeito em litros por milha náutica (ajustável na interface)
CONSUMO_LITROS_NM_PADRAO = 8.0

# Classificação das áreas de ocorrência do GAMA
AREAS_MARITIMAS = {
    'Territorial sea',
    'High sea - Within EEZ',
    'High sea - Outside EEZ',
    'High sea - n/a',
    'High sea',
}
AREAS_TERRESTRES = {
    'Internal waters - Port area',
    'Internal waters - Channel; river',
    'Internal waters - Other',
    'Internal waters - Archipelago fairway',
    'Inland waters - River',
    'Inland waters - Channel',
    'Inland waters - Lake',
    'Inland waters - Other',
    'Repair yard',
    'Unknown',
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. GEOMETRIA DA COSTA
# ═══════════════════════════════════════════════════════════════════════════
# Simplificação para apoio à decisão. Para uso operacional/legal, substituir
# por coordenadas oficiais (DGRM / IHPT / Decreto-Lei das águas marítimas).

# Portugal continental — sequência (lon, lat) norte → sul
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

# Açores — contornos exteriores das ilhas (loop fechado, lon, lat)
ACORES_CONTORNOS = [
    [(-31.27, 39.51), (-31.19, 39.37), (-31.08, 39.37), (-31.00, 39.51), (-31.27, 39.51)],   # Flores
    [(-31.13, 39.77), (-31.08, 39.68), (-31.19, 39.63), (-31.23, 39.72), (-31.13, 39.77)],   # Corvo
    [(-28.83, 38.62), (-28.75, 38.51), (-28.62, 38.50), (-28.55, 38.60), (-28.70, 38.68), (-28.83, 38.62)],  # Faial
    [(-28.53, 38.55), (-28.10, 38.38), (-27.97, 38.42), (-28.08, 38.57), (-28.53, 38.55)],   # Pico
    [(-28.26, 38.77), (-27.90, 38.63), (-27.75, 38.65), (-28.07, 38.78), (-28.26, 38.77)],   # São Jorge
    [(-28.10, 39.10), (-27.97, 39.02), (-27.93, 39.07), (-28.05, 39.15), (-28.10, 39.10)],   # Graciosa
    [(-27.40, 38.80), (-27.15, 38.65), (-26.97, 38.72), (-27.23, 38.83), (-27.40, 38.80)],   # Terceira
    [(-25.85, 37.88), (-25.47, 37.70), (-25.13, 37.73), (-25.12, 37.86), (-25.52, 37.93), (-25.85, 37.88)],  # São Miguel
    [(-25.23, 37.02), (-25.03, 36.93), (-24.87, 36.97), (-25.00, 37.08), (-25.23, 37.02)],   # Santa Maria
]

# Madeira — ilha principal, Porto Santo, Desertas, Selvagens
MADEIRA_CONTORNOS = [
    [(-17.27, 32.90), (-17.00, 32.62), (-16.68, 32.62), (-16.30, 32.78), (-16.68, 32.90), (-17.27, 32.90)],  # Madeira
    [(-16.43, 33.12), (-16.28, 33.03), (-16.25, 33.08), (-16.38, 33.13), (-16.43, 33.12)],  # Porto Santo
    [(-16.52, 32.57), (-16.48, 32.38), (-16.43, 32.50), (-16.52, 32.57)],                   # Desertas
    [(-15.92, 30.18), (-15.83, 30.10), (-15.80, 30.15), (-15.92, 30.18)],                   # Selvagens
]

# Trechos da costa para correção de pontos costeiros
_COSTA_OESTE = COSTA_PONTOS[:18]
_COSTA_SUL   = COSTA_PONTOS[17:]


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONSTRUÇÃO DAS ZONAS DE PATRULHA
# ═══════════════════════════════════════════════════════════════════════════

def _factory_projecao(lat_ref):
    """Projeção/desprojeção equirretangular local para uma latitude de referência."""
    m_por_grau_lon = _M_POR_GRAU_LAT * cos(radians(lat_ref))

    def proj(x, y, z=None):
        return (x * m_por_grau_lon, y * _M_POR_GRAU_LAT)

    def desproj(x, y, z=None):
        return (x / m_por_grau_lon, y / _M_POR_GRAU_LAT)

    return proj, desproj


def _buffer_geografico(geom, dist_nm, lat_ref):
    """Buffer em milhas náuticas sobre uma geometria geográfica (EPSG:4326).

    Converte para metros via projeção equirretangular local, aplica o buffer
    e regressa a graus. Precisão tipicamente < 1 % para as extensões usadas.
    """
    proj, desproj = _factory_projecao(lat_ref)
    buf_proj = transform(proj, geom).buffer(dist_nm * NM_EM_METROS)
    return transform(desproj, buf_proj)


def _construir_zonas():
    """Gera as 6 faixas (anéis) de distância à costa por buffers sucessivos.

    Cada região (Continente, Açores, Madeira) usa a sua própria latitude de
    referência para minimizar a distorção. Os buffers são depois unidos numa
    geometria única por distância.
    """
    # Buffers por região e distância
    buffers_total = {}
    for d in LIMITES_NM:
        partes = [_buffer_geografico(COSTA, d, 39.0)]
        for g in [LineString(c) for c in ACORES_CONTORNOS]:
            partes.append(_buffer_geografico(g, d, 38.5))
        for g in [LineString(c) for c in MADEIRA_CONTORNOS]:
            partes.append(_buffer_geografico(g, d, 32.0))
        buffers_total[d] = unary_union(partes)

    # Massa terrestre: continente + ilhas
    norte, sul = COSTA_PONTOS[0], COSTA_PONTOS[-1]
    massa_continente = Polygon(COSTA_PONTOS + [(2.5, sul[1]), (2.5, norte[1])])
    ilhas_terra = [
        Polygon(c) for c in ACORES_CONTORNOS + MADEIRA_CONTORNOS if len(c) >= 3
    ]
    massa_terrestre = (
        unary_union([massa_continente] + ilhas_terra) if ilhas_terra else massa_continente
    )

    # Área de interesse: cobre toda a ZEE portuguesa
    # (Açores a oeste ~-35°, Selvagens a sul ~29.5°, continente a leste ~-6°)
    area_interesse = box(-36.0, 29.0, -6.0, 43.5)

    # Anéis por diferença entre buffers consecutivos
    fronteiras = [0] + LIMITES_NM
    zonas = {}
    for i, (inferior, superior) in enumerate(zip(fronteiras[:-1], fronteiras[1:]), start=1):
        anel = (
            buffers_total[superior] if inferior == 0
            else buffers_total[superior].difference(buffers_total[inferior])
        )
        zonas[i] = anel.intersection(area_interesse).difference(massa_terrestre)

    zonas[6] = (
        area_interesse
        .difference(buffers_total[LIMITES_NM[-1]])
        .difference(massa_terrestre)
    )
    return zonas, massa_terrestre


# Construídos uma única vez no arranque do módulo
ZONAS_POLIGONOS, MASSA_TERRESTRE = _construir_zonas()


# ═══════════════════════════════════════════════════════════════════════════
# 4. DISTÂNCIAS E CLASSIFICAÇÃO GEOGRÁFICA
# ═══════════════════════════════════════════════════════════════════════════

def haversine_km(lat1, lon1, lat2, lon2):
    """Distância em km entre dois pontos geográficos (fórmula de Haversine)."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def distancia_costa_km(lat, lon):
    """Distância em km à linha de costa portuguesa mais próxima.

    Considera o continente e os contornos dos arquipélagos, evitando
    referenciar ocorrências insulares à costa continental.
    """
    ponto = Point(lon, lat)
    p_cont, _ = nearest_points(COSTA, ponto)
    melhor = haversine_km(lat, lon, p_cont.y, p_cont.x)

    for contorno in ACORES_CONTORNOS + MADEIRA_CONTORNOS:
        p_prox, _ = nearest_points(LineString(contorno), ponto)
        d = haversine_km(lat, lon, p_prox.y, p_prox.x)
        if d < melhor:
            melhor = d

    return melhor


def distancia_costa_nm(lat, lon):
    """Distância em milhas náuticas à costa portuguesa mais próxima."""
    return distancia_costa_km(lat, lon) / 1.852


def zona_por_distancia_nm(distancia_nm):
    """Zona de patrulha (1–6) correspondente a uma distância à costa em NM."""
    for limite, zona in zip(LIMITES_NM, range(1, len(LIMITES_NM) + 1)):
        if distancia_nm <= limite:
            return zona
    return 6  # alto mar


def zona_atual_navio(pos_navio):
    """Zona de patrulha (1–6) da posição actual do navio."""
    lat, lon = pos_navio
    return zona_por_distancia_nm(distancia_costa_nm(lat, lon))


def distancia_a_poligono(pos_navio, poligono):
    """Distância em km de um ponto (lat, lon) ao polígono de uma zona."""
    lat, lon = pos_navio
    ponto = Point(lon, lat)
    if poligono is None or poligono.is_empty:
        return float('nan')
    if poligono.contains(ponto):
        return 0.0
    p_proximo, _ = nearest_points(poligono, ponto)
    return haversine_km(lat, lon, p_proximo.y, p_proximo.x)


def calcular_distancias(pos_navio, zonas=ZONAS_POLIGONOS):
    """Dicionário {zona: distância_km} do navio a cada zona de patrulha."""
    return {z: distancia_a_poligono(pos_navio, poly) for z, poly in zonas.items()}


def identificar_teatro_maritimo(lat, lon):
    """Teatro marítimo português da coordenada: 'Continente', 'Açores', 'Madeira' ou None."""
    lat, lon = float(lat), float(lon)
    if 34.5 <= lat <= 43.5 and -16.0 <= lon <= -6.0:
        return "Continente"
    if 33.0 <= lat <= 43.0 and -36.0 <= lon <= -21.0:
        return "Açores"
    if 28.5 <= lat <= 36.0 and -20.0 <= lon <= -12.0:
        return "Madeira"
    return None


def em_aguas_portugal(lat, lon):
    """True se a coordenada estiver dentro de um espaço marítimo português."""
    return identificar_teatro_maritimo(lat, lon) is not None


def ponto_em_terra(lat, lon):
    """True se o ponto estiver na massa terrestre (continente ou ilhas)."""
    return MASSA_TERRESTRE.contains(Point(lon, lat))


# Alias mantido por compatibilidade
ponto_em_mar_valido = em_aguas_portugal


# ═══════════════════════════════════════════════════════════════════════════
# 5. CORREÇÃO DE COORDENADAS COSTEIRAS
# ═══════════════════════════════════════════════════════════════════════════
# O GAMA tem registos marítimos cujas coordenadas aparecem ligeiramente em
# terra. Em vez de os apagar, o ponto é deslocado para o mar.

def _interp_lon_costa_oeste(lat):
    """Longitude interpolada da costa oeste para uma dada latitude."""
    candidatos = []
    for (lon1, lat1), (lon2, lat2) in zip(_COSTA_OESTE, _COSTA_OESTE[1:]):
        if min(lat1, lat2) <= lat <= max(lat1, lat2) and abs(lat2 - lat1) > 1e-9:
            t = (lat - lat1) / (lat2 - lat1)
            candidatos.append(lon1 + t * (lon2 - lon1))
    return float(np.median(candidatos)) if candidatos else None


def _interp_lat_costa_sul(lon):
    """Latitude interpolada da costa sul para uma dada longitude."""
    candidatos = []
    for (lon1, lat1), (lon2, lat2) in zip(_COSTA_SUL, _COSTA_SUL[1:]):
        if min(lon1, lon2) <= lon <= max(lon1, lon2) and abs(lon2 - lon1) > 1e-9:
            t = (lon - lon1) / (lon2 - lon1)
            candidatos.append(lat1 + t * (lat2 - lat1))
    return float(np.median(candidatos)) if candidatos else None


def _ponto_no_lado_maritimo_continental(lat, lon, margem_graus=0.003):
    """True se a coordenada já está do lado do mar no continente."""
    if not (35.0 <= lat <= 41.95 and -13.5 <= lon <= -6.5):
        return False
    # Algarve: mar fica a sul
    if lat < 37.20 and lon > -8.90:
        lat_costa = _interp_lat_costa_sul(lon)
        return lat_costa is None or lat <= lat_costa - margem_graus
    # Costa oeste: mar fica a oeste
    lon_costa = _interp_lon_costa_oeste(lat)
    return lon_costa is None or lon <= lon_costa - margem_graus


def corrigir_ponto_para_mar(lat, lon, deslocamento_graus=0.035):
    """Desloca visualmente um ponto costeiro para o lado do mar.

    Só actua em pontos do continente que apareçam em terra. Não elimina
    o registo — apenas corrige a posição de visualização.
    """
    lat, lon = float(lat), float(lon)
    if identificar_teatro_maritimo(lat, lon) != "Continente":
        return lat, lon
    if _ponto_no_lado_maritimo_continental(lat, lon):
        return lat, lon

    p_costa, _ = nearest_points(COSTA, Point(lon, lat))
    # Algarve: empurra para sul
    if p_costa.y < 37.25 and p_costa.x > -8.95:
        return p_costa.y - deslocamento_graus, p_costa.x
    # Costa oeste: empurra para oeste
    return p_costa.y, p_costa.x - deslocamento_graus


# ═══════════════════════════════════════════════════════════════════════════
# 6. LEITURA E NORMALIZAÇÃO DO CSV GAMA
# ═══════════════════════════════════════════════════════════════════════════

def parse_coordenada_dms(texto, eixo='lat'):
    """Converte '38°41.08' N' ou '9°0.03' W' para graus decimais."""
    if pd.isna(texto):
        return None
    s = re.sub(r'["""]', '', str(texto).strip())
    padrao = (
        r"(\d+)°(\d+(?:\.\d+)?)'?\s*([NS])" if eixo == 'lat'
        else r"(\d+)°(\d+(?:\.\d+)?)'?\s*([EW])"
    )
    m = re.match(padrao, s)
    if not m:
        return None
    graus, minutos, hem = float(m.group(1)), float(m.group(2)), m.group(3)
    dec = graus + minutos / 60.0
    return -dec if hem in ('S', 'W') else dec


def _importancia_de_severidade(severidade):
    """Valor numérico de importância a partir do campo Occurrence severity."""
    return SEVERIDADE_IMPORTANCIA.get(str(severidade).strip(), 5.0)


def _marcar_acidente(row):
    """Retorna 1 se a ocorrência for considerada um acidente grave."""
    vidas = pd.to_numeric(row.get('Lives lost Occurrence-Total', 0), errors='coerce') or 0
    if vidas > 0:
        return 1
    if str(row.get('Did the ship sink?', '')).strip().lower() == 'yes':
        return 1
    if str(row.get('Occurrence severity', '')).strip() in ('Very serious', 'Serious'):
        return 1
    navio = str(row.get('Occurrence with ship(s)', '')).lower()
    if any(k in navio for k in ('collision', 'grounding', 'foundering', 'fire/explosion', 'flooding')):
        return 1
    return 0


def filtrar_incidentes_maritimos(df):
    """Filtra o DataFrame bruto do GAMA para manter só ocorrências marítimas.

    Remove portos, rios, águas interiores e coordenadas fora das janelas
    geográficas de Portugal. Pontos ligeiramente em terra são corrigidos,
    não eliminados.
    """
    df = df.copy()

    # Converter coordenadas DMS → decimal, se necessário
    if 'Lat' not in df.columns:
        df['Lat'] = df.get('Latitude', pd.Series(dtype=float)).apply(
            lambda x: parse_coordenada_dms(x, 'lat')
        )
    if 'Lon' not in df.columns:
        df['Lon'] = df.get('Longitude', pd.Series(dtype=float)).apply(
            lambda x: parse_coordenada_dms(x, 'lon')
        )

    df = df.dropna(subset=['Lat', 'Lon']).copy()
    df['Lat_original'] = df['Lat'].astype(float)
    df['Lon_original'] = df['Lon'].astype(float)

    # Filtrar por tipo de área
    if 'Sea area of occurrence' in df.columns:
        df = df[~df['Sea area of occurrence'].isin(AREAS_TERRESTRES)]
        df = df[df['Sea area of occurrence'].isin(AREAS_MARITIMAS)]

    # Filtrar por teatro marítimo português
    df['Teatro_Maritimo'] = [
        identificar_teatro_maritimo(lat, lon)
        for lat, lon in zip(df['Lat_original'], df['Lon_original'])
    ]
    df = df[df['Teatro_Maritimo'].notna()].copy()

    # Corrigir pontos costeiros deslocados para terra
    corrigidos = [
        corrigir_ponto_para_mar(lat, lon)
        for lat, lon in zip(df['Lat_original'], df['Lon_original'])
    ]
    if corrigidos:
        df['Lat'] = [p[0] for p in corrigidos]
        df['Lon'] = [p[1] for p in corrigidos]

    return df.reset_index(drop=True)


def normalizar_incidentes_pontuais(df_bruto):
    """Converte export GAMA num DataFrame marítimo padronizado.

    Devolve colunas: Lat, Lon, Importancia, Acidente, Teatro_Maritimo,
    Lat_original, Lon_original.
    """
    mar = filtrar_incidentes_maritimos(df_bruto)

    if 'Importancia' not in mar.columns:
        mar['Importancia'] = (
            mar['Occurrence severity'].apply(_importancia_de_severidade)
            if 'Occurrence severity' in mar.columns
            else 5.0
        )
    if 'Acidente' not in mar.columns:
        mar['Acidente'] = mar.apply(_marcar_acidente, axis=1).astype(int)

    cols = ['Lat', 'Lon', 'Importancia', 'Acidente']
    for extra in ['Teatro_Maritimo', 'Lat_original', 'Lon_original']:
        if extra in mar.columns:
            cols.append(extra)

    return mar[cols].copy()


def carregar_incidentes_gama(caminho_csv):
    """Lê o export GAMA (CSV) e devolve DataFrame marítimo normalizado."""
    return normalizar_incidentes_pontuais(pd.read_csv(caminho_csv))


def exportar_incidentes_maritimos(caminho_entrada, caminho_saida):
    """Gera CSV limpo (só mar) a partir do export GAMA original.

    Devolve (n_total, n_maritimos).
    """
    bruto = pd.read_csv(caminho_entrada)
    mar = filtrar_incidentes_maritimos(bruto)
    mar.to_csv(caminho_saida, index=False)
    return len(bruto), len(mar)


# ═══════════════════════════════════════════════════════════════════════════
# 7. AGREGAÇÃO POR ZONA
# ═══════════════════════════════════════════════════════════════════════════

def atribuir_zonas_pontuais(df_pontos):
    """Acrescenta Distancia_Costa_NM e Zona_Patrulha a cada incidente."""
    df = df_pontos.copy()
    df['Distancia_Costa_NM'] = [
        distancia_costa_nm(lat, lon) for lat, lon in zip(df['Lat'], df['Lon'])
    ]
    df['Zona_Patrulha'] = df['Distancia_Costa_NM'].apply(zona_por_distancia_nm)
    return df


def agregar_por_zona(df_pontos_com_zona):
    """Agrega incidentes por zona: contagem, gravidade média e nº de acidentes.

    Garante que todas as zonas 1–6 estão presentes, mesmo com zero ocorrências.
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
    agg = agg.astype({'Zona_Patrulha': int, 'Num_Incidentes': int, 'Acidentes_Ultimo_Ano': int})
    return agg


# ═══════════════════════════════════════════════════════════════════════════
# 8. MOTOR DE PONTUAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def score_incidentes_com_decaimento(datas_incidentes, lambda_anual=0.3,
                                    hoje: datetime | None = None):
    """Score ponderado por decaimento exponencial temporal (incidentes mais
    recentes têm maior peso)."""
    if hoje is None:
        hoje = datetime.now()
    if not datas_incidentes:
        return 0.0
    idades = np.array([(hoje - d).days / 365.25 for d in datas_incidentes])
    return float(np.sum(np.exp(-lambda_anual * idades)))


def _percentil(serie, inverter=False):
    """Normaliza por rank percentil para [0, 1].

    Cada valor recebe a sua posição relativa entre as zonas (0 = mínimo,
    1 = máximo), tornando a pontuação robusta a outliers e estável entre
    actualizações de dados. Em caso de empate, usa a média dos ranks.
    Quando todos os valores são iguais, devolve 0.5 para não privilegiar
    nenhuma zona.
    """
    s = serie.astype(float)
    if s.nunique() == 1:
        return pd.Series(0.5, index=s.index)
    ranked = s.rank(method='average')          # empates → média dos ranks
    norm = (ranked - ranked.min()) / (ranked.max() - ranked.min())
    return 1 - norm if inverter else norm


def _minmax(serie, inverter=False):
    """Normaliza por Min-Max para [0, 1] — usado apenas para a distância.

    A distância é contínua e sem outliers extremos, pelo que a magnitude
    relativa importa e o Min-Max é adequado.
    """
    s = serie.astype(float)
    minimo, maximo = s.min(), s.max()
    if maximo == minimo:
        return pd.Series(0.5, index=s.index)
    norm = (s - minimo) / (maximo - minimo)
    return 1 - norm if inverter else norm


def preparar_dataframe(incidentes, distancias):
    """Acrescenta colunas normalizadas ao DataFrame de incidentes por zona.

    Incidentes, gravidade e acidentes → rank percentil (robusto a outliers).
    Distância → Min-Max (magnitude contínua sem outliers extremos).
    """
    df = incidentes.copy()
    df['Zona_Patrulha'] = df['Zona_Patrulha'].astype(int)
    df['Distancia'] = df['Zona_Patrulha'].map(distancias)
    df['Num_Incidentes_norm']       = _percentil(df['Num_Incidentes'])
    df['Importancia_norm']          = _percentil(df['Importancia'])
    df['Acidentes_Ultimo_Ano_norm'] = _percentil(df['Acidentes_Ultimo_Ano'])
    df['Distancia_norm']            = _minmax(df['Distancia'], inverter=True)
    return df


def calcular_pontuacao(df, pesos):
    """Pontuação ponderada de cada zona de acordo com o perfil de pesos."""
    return (
        pesos['incidentes'] * df['Num_Incidentes_norm'] +
        pesos['gravidade']  * df['Importancia_norm'] +
        pesos['acidentes']  * df['Acidentes_Ultimo_Ano_norm'] +
        pesos['distancia']  * df['Distancia_norm']
    )


def gerar_justificação(df, distancias, pesos, top_k=3):
    """Gera justificações textuais para as top_k zonas recomendadas.

    Identifica o critério dominante e assinala decisões apertadas (< 5 %).
    """
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

    justificações = []
    for i in range(min(top_k, len(work))):
        row = work.iloc[i]
        zona = int(row['Zona_Patrulha'])
        contribs = {k: row[f'contrib_{k}'] for k in criterios}
        dominante = max(contribs, key=contribs.get)
        peso_dom = contribs[dominante] / row['Pontuacao'] if row['Pontuacao'] else 0
        _, label_dom = criterios[dominante]

        alerta = ""
        if i + 1 < len(work):
            diff = row['Pontuacao'] - work.iloc[i + 1]['Pontuacao']
            if row['Pontuacao'] and diff / row['Pontuacao'] < 0.05:
                alerta = " ⚠️ decisão apertada"

        justificações.append({
            'posicao':            i + 1,
            'zona':               zona,
            'pontuacao':          row['Pontuacao'],
            'criterio_dominante': label_dom,
            'peso_dominante':     peso_dom,
            'distancia':          distancias[zona],
            'incidentes':         int(row['Num_Incidentes']),
            'gravidade':          row['Importancia'],
            'acidentes':          int(row['Acidentes_Ultimo_Ano']),
            'alerta':             alerta,
        })

    return justificações


# ═══════════════════════════════════════════════════════════════════════════
# 9. AUTONOMIA E NAVEGAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def calcular_autonomia(combustivel_litros, consumo_litros_nm=CONSUMO_LITROS_NM_PADRAO):
    """Autonomia e raio de ida-e-volta a partir do combustível disponível.

    Devolve dict com alcance_total_nm/km e raio_ida_volta_nm/km.
    """
    if combustivel_litros <= 0 or consumo_litros_nm <= 0:
        return {
            'alcance_total_nm':    0.0,
            'alcance_total_km':    0.0,
            'raio_ida_volta_nm':   0.0,
            'raio_ida_volta_km':   0.0,
        }
    alcance_nm = combustivel_litros / consumo_litros_nm
    raio_nm    = alcance_nm / 2.0
    return {
        'alcance_total_nm':    alcance_nm,
        'alcance_total_km':    alcance_nm * 1.852,
        'raio_ida_volta_nm':   raio_nm,
        'raio_ida_volta_km':   raio_nm * 1.852,
    }


def circulo_autonomia(lat, lon, raio_km, n_pontos=72):
    """Polígono (lon, lat) aproximando um círculo geodésico de raio raio_km."""
    if raio_km <= 0:
        return []
    R = 6371.0
    lat_r, lon_r = radians(lat), radians(lon)
    ang_dist = raio_km / R
    pontos = []
    for i in range(n_pontos + 1):
        bearing = radians(i * 360.0 / n_pontos)
        lat2 = asin(
            sin(lat_r) * cos(ang_dist) +
            cos(lat_r) * sin(ang_dist) * cos(bearing)
        )
        lon2 = lon_r + atan2(
            sin(bearing) * sin(ang_dist) * cos(lat_r),
            cos(ang_dist) - sin(lat_r) * sin(lat2),
        )
        pontos.append((degrees(lon2), degrees(lat2)))
    return pontos
