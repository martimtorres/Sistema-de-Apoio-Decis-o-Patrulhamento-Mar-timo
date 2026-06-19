"""
Sistema de Apoio à Decisão para Patrulhamento Marítimo — núcleo.

Inclui: perfis de pesos, decaimento temporal, e divisão do mar em faixas
de distância à costa portuguesa (Z1..Z6), construídas por buffers
sucessivos à linha de costa — não por polígonos desenhados manualmente.
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt, degrees, atan2
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

# ── Linhas de costa simplificadas dos arquipélagos (lon, lat).
# Cada arquipélago é representado pela sua costa exterior (loop fechado),
# o que permite gerar buffers em metros a partir do contorno real de cada ilha.
# São simplificações para apoio à decisão — não substitui cartografia oficial.

# Açores: grupo ocidental, central e oriental
ACORES_CONTORNOS = [
    # Flores
    [(-31.27, 39.51), (-31.19, 39.37), (-31.08, 39.37), (-31.00, 39.51), (-31.27, 39.51)],
    # Corvo
    [(-31.13, 39.77), (-31.08, 39.68), (-31.19, 39.63), (-31.23, 39.72), (-31.13, 39.77)],
    # Faial
    [(-28.83, 38.62), (-28.75, 38.51), (-28.62, 38.50), (-28.55, 38.60), (-28.70, 38.68), (-28.83, 38.62)],
    # Pico
    [(-28.53, 38.55), (-28.10, 38.38), (-27.97, 38.42), (-28.08, 38.57), (-28.53, 38.55)],
    # São Jorge
    [(-28.26, 38.77), (-27.90, 38.63), (-27.75, 38.65), (-28.07, 38.78), (-28.26, 38.77)],
    # Graciosa
    [(-28.10, 39.10), (-27.97, 39.02), (-27.93, 39.07), (-28.05, 39.15), (-28.10, 39.10)],
    # Terceira
    [(-27.40, 38.80), (-27.15, 38.65), (-26.97, 38.72), (-27.23, 38.83), (-27.40, 38.80)],
    # São Miguel
    [(-25.85, 37.88), (-25.47, 37.70), (-25.13, 37.73), (-25.12, 37.86), (-25.52, 37.93), (-25.85, 37.88)],
    # Santa Maria
    [(-25.23, 37.02), (-25.03, 36.93), (-24.87, 36.97), (-25.00, 37.08), (-25.23, 37.02)],
]

# Madeira: ilha principal, Porto Santo, Desertas, Selvagens
MADEIRA_CONTORNOS = [
    # Madeira
    [(-17.27, 32.90), (-17.00, 32.62), (-16.68, 32.62), (-16.30, 32.78), (-16.68, 32.90), (-17.27, 32.90)],
    # Porto Santo
    [(-16.43, 33.12), (-16.28, 33.03), (-16.25, 33.08), (-16.38, 33.13), (-16.43, 33.12)],
    # Desertas
    [(-16.52, 32.57), (-16.48, 32.38), (-16.43, 32.50), (-16.52, 32.57)],
    # Selvagens
    [(-15.92, 30.18), (-15.83, 30.10), (-15.80, 30.15), (-15.92, 30.18)],
]

# ── Pontos de referência das ilhas (usados para cálculo de distância mínima)
ILHAS_PT_PONTOS = [
    # Açores
    (-31.13, 39.45),  # Flores
    (-31.11, 39.70),  # Corvo
    (-28.70, 38.58),  # Faial
    (-28.33, 38.47),  # Pico
    (-28.05, 38.65),  # São Jorge
    (-28.02, 39.05),  # Graciosa
    (-27.22, 38.72),  # Terceira
    (-25.47, 37.78),  # São Miguel
    (-25.10, 36.97),  # Santa Maria
    # Madeira
    (-16.95, 32.75),  # Madeira
    (-16.35, 33.05),  # Porto Santo
    (-16.50, 32.50),  # Desertas
    (-15.87, 30.15),  # Selvagens
]
ILHAS_PT = [Point(lon, lat) for lon, lat in ILHAS_PT_PONTOS]

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

_M_POR_GRAU_LAT = 111_320.0


def _factory_projecao(lat_ref):
    """Devolve funções de projeção/desprojeção equirretangular para uma latitude de referência."""
    m_por_grau_lon = _M_POR_GRAU_LAT * cos(radians(lat_ref))

    def proj(x, y, z=None):
        return (x * m_por_grau_lon, y * _M_POR_GRAU_LAT)

    def desproj(x, y, z=None):
        return (x / m_por_grau_lon, y / _M_POR_GRAU_LAT)

    return proj, desproj


def _buffer_geografico(geom, dist_nm, lat_ref):
    """Buffer em milhas náuticas numa geometria geográfica (EPSG:4326).

    Usa projeção equirretangular local centrada em lat_ref para converter
    a geometria para metros, faz o buffer, e regressa a graus.
    A precisão é tipicamente < 1 % para as extensões aqui usadas.
    """
    proj, desproj = _factory_projecao(lat_ref)
    geom_proj = transform(proj, geom)
    buf_proj = geom_proj.buffer(dist_nm * NM_EM_METROS)
    return transform(desproj, buf_proj)


def _construir_zonas():
    """Gera as 6 faixas (anéis) de distância à costa por buffers sucessivos.

    Cada arquipélago (Continente, Açores, Madeira) é projetado com a sua
    própria latitude de referência para minimizar a distorção dos buffers.
    Os resultados são reunidos numa geometria única por distância, cobrindo
    assim toda a extensão marítima portuguesa.
    """
    from shapely.ops import unary_union

    # ── 1. Linha de costa do Continente
    lat_ref_cont = 39.0
    costa_geom = COSTA

    # ── 2. Contornos dos Açores (grupo de polígonos/linhas de costa)
    lat_ref_acores = 38.5
    acores_geoms = [LineString(c) for c in ACORES_CONTORNOS]

    # ── 3. Contornos da Madeira
    lat_ref_madeira = 32.0
    madeira_geoms = [LineString(c) for c in MADEIRA_CONTORNOS]

    # ── 4. Buffer por distância para cada grupo, depois une tudo
    buffers_total = {}
    for d in LIMITES_NM:
        partes = [_buffer_geografico(costa_geom, d, lat_ref_cont)]
        for g in acores_geoms:
            partes.append(_buffer_geografico(g, d, lat_ref_acores))
        for g in madeira_geoms:
            partes.append(_buffer_geografico(g, d, lat_ref_madeira))
        buffers_total[d] = unary_union(partes)

    # ── 5. Massa terrestre: continente + ilhas
    norte, sul = COSTA_PONTOS[0], COSTA_PONTOS[-1]
    massa_continente = Polygon(COSTA_PONTOS + [(2.5, sul[1]), (2.5, norte[1])])

    ilhas_terra = []
    for contorno in ACORES_CONTORNOS + MADEIRA_CONTORNOS:
        if len(contorno) >= 3:
            ilhas_terra.append(Polygon(contorno))
    if ilhas_terra:
        massa_ilhas = unary_union(ilhas_terra)
        massa_terrestre = unary_union([massa_continente, massa_ilhas])
    else:
        massa_terrestre = massa_continente

    # ── 6. Área de interesse: cobre toda a ZEE portuguesa
    # Açores mais a oeste: ~-35°; Selvagens mais a sul: ~29.5°; continente a leste: ~-6°
    area_interesse = box(-36.0, 29.0, -6.0, 43.5)

    # ── 7. Construção dos anéis (diferenças entre buffers sucessivos)
    fronteiras = [0] + LIMITES_NM
    zonas = {}
    for i, (inferior, superior) in enumerate(zip(fronteiras[:-1], fronteiras[1:]), start=1):
        anel = (
            buffers_total[superior]
            if inferior == 0
            else buffers_total[superior].difference(buffers_total[inferior])
        )
        zonas[i] = anel.intersection(area_interesse).difference(massa_terrestre)

    zonas[6] = (
        area_interesse
        .difference(buffers_total[LIMITES_NM[-1]])
        .difference(massa_terrestre)
    )
    return zonas, massa_terrestre


ZONAS_POLIGONOS, MASSA_TERRESTRE = _construir_zonas()

# Áreas marítimas válidas (exclui portos, rios, águas interiores e estaleiros).
AREAS_MARITIMAS = {
    'Territorial sea', 'High sea - Within EEZ', 'High sea - Outside EEZ',
    'High sea - n/a', 'High sea',
}
AREAS_TERRESTRES = {
    'Internal waters - Port area', 'Internal waters - Channel; river',
    'Internal waters - Other', 'Internal waters - Archipelago fairway',
    'Inland waters - River', 'Inland waters - Channel', 'Inland waters - Lake',
    'Inland waters - Other', 'Repair yard', 'Unknown',
}

# Consumo por defeito: litros por milha náutica (ajustável na interface).
CONSUMO_LITROS_NM_PADRAO = 8.0

SEVERIDADE_IMPORTANCIA = {
    'Very serious': 10.0,
    'Serious': 8.0,
    'Less Serious': 5.0,
    'Marine incident': 4.0,
}


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
    p_proximo, _ = nearest_points(poligono, ponto)
    return haversine_km(lat, lon, p_proximo.y, p_proximo.x)


def calcular_distancias(pos_navio, zonas=ZONAS_POLIGONOS):
    return {z: distancia_a_poligono(pos_navio, poly) for z, poly in zonas.items()}


def distancia_costa_km(lat, lon):
    """Distância (km) à linha de costa portuguesa mais próxima.

    Usa a linha de costa do continente e os contornos das ilhas dos Açores
    e da Madeira, o que permite classificar corretamente ocorrências nesses
    arquipélagos sem as referenciar à costa continental.
    """
    ponto = Point(lon, lat)

    # Continente
    p_cont, _ = nearest_points(COSTA, ponto)
    melhor = haversine_km(lat, lon, p_cont.y, p_cont.x)

    # Ilhas — usa os contornos (LineString) para maior precisão
    for contorno in ACORES_CONTORNOS + MADEIRA_CONTORNOS:
        linha = LineString(contorno)
        p_prox, _ = nearest_points(linha, ponto)
        d = haversine_km(lat, lon, p_prox.y, p_prox.x)
        if d < melhor:
            melhor = d

    return melhor


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


def ponto_em_terra(lat, lon):
    """True se o ponto estiver a leste da linha de costa (continente)."""
    return MASSA_TERRESTRE.contains(Point(lon, lat))


def identificar_teatro_maritimo(lat, lon):
    """Identifica se a coordenada cai num dos espaços marítimos portugueses.

    A filtragem é feita por janelas geográficas largas para manter todas as
    ocorrências marítimas portuguesas do Continente, Açores e Madeira, sem
    apagar pontos de zonas mais afastadas. Registos globais reportados por
    Portugal, mas fora destas áreas, ficam excluídos.
    """
    lat = float(lat)
    lon = float(lon)

    # Portugal continental e mar adjacente (ZEE até 200 NM ≈ ~370 km ≈ ~3.3° lat/lon)
    if 34.5 <= lat <= 43.5 and -16.0 <= lon <= -6.0:
        return "Continente"

    # Açores e mar adjacente (ZEE até 200 NM em torno dos grupos)
    if 33.0 <= lat <= 43.0 and -36.0 <= lon <= -21.0:
        return "Açores"

    # Madeira, Porto Santo, Desertas e Selvagens (ZEE até 200 NM)
    if 28.5 <= lat <= 36.0 and -20.0 <= lon <= -12.0:
        return "Madeira"

    return None


def em_aguas_portugal(lat, lon):
    """True se a coordenada estiver numa janela marítima portuguesa."""
    return identificar_teatro_maritimo(lat, lon) is not None


# Trechos da costa usados para corrigir visualmente coordenadas costeiras que
# vêm registadas como "mar", mas aparecem ligeiramente em terra no mapa.
_COSTA_OESTE = COSTA_PONTOS[:18]
_COSTA_SUL = COSTA_PONTOS[17:]


def _interp_lon_costa_oeste(lat):
    """Longitude aproximada da costa oeste para uma dada latitude."""
    candidatos = []
    for (lon1, lat1), (lon2, lat2) in zip(_COSTA_OESTE, _COSTA_OESTE[1:]):
        if min(lat1, lat2) <= lat <= max(lat1, lat2) and abs(lat2 - lat1) > 1e-9:
            t = (lat - lat1) / (lat2 - lat1)
            candidatos.append(lon1 + t * (lon2 - lon1))
    if not candidatos:
        return None
    return float(np.median(candidatos))


def _interp_lat_costa_sul(lon):
    """Latitude aproximada da costa sul para uma dada longitude."""
    candidatos = []
    for (lon1, lat1), (lon2, lat2) in zip(_COSTA_SUL, _COSTA_SUL[1:]):
        if min(lon1, lon2) <= lon <= max(lon1, lon2) and abs(lon2 - lon1) > 1e-9:
            t = (lon - lon1) / (lon2 - lon1)
            candidatos.append(lat1 + t * (lat2 - lat1))
    if not candidatos:
        return None
    return float(np.median(candidatos))


def _ponto_no_lado_maritimo_continental(lat, lon, margem_graus=0.003):
    """Verificação simples de lado mar/terra no continente.

    Serve só para decidir se uma coordenada deve ser ajustada visualmente para
    o mar. Não é usada para apagar ocorrências portuguesas.
    """
    if not (35.0 <= lat <= 41.95 and -13.5 <= lon <= -6.5):
        return False

    # A sul de Sagres/Algarve, o mar fica principalmente para sul da costa.
    if lat < 37.20 and lon > -8.90:
        lat_costa = _interp_lat_costa_sul(lon)
        if lat_costa is None:
            return True
        return lat <= lat_costa - margem_graus

    # Costa oeste: o mar fica para oeste da costa.
    lon_costa = _interp_lon_costa_oeste(lat)
    if lon_costa is None:
        return True
    return lon <= lon_costa - margem_graus


def corrigir_ponto_para_mar(lat, lon, deslocamento_graus=0.035):
    """Corrige apenas a posição de visualização de pontos costeiros.

    O ficheiro GAMA tem alguns registos classificados como "Territorial sea" ou
    "High sea - Within EEZ" cujas coordenadas aparecem uns quilómetros em terra
    por aproximação/erro de registo. Em vez de apagar esses incidentes, o ponto
    é encostado à costa e deslocado ligeiramente para o lado do mar.
    """
    lat = float(lat)
    lon = float(lon)
    teatro = identificar_teatro_maritimo(lat, lon)
    if teatro != "Continente":
        return lat, lon

    if _ponto_no_lado_maritimo_continental(lat, lon):
        return lat, lon

    ponto = Point(lon, lat)
    p_costa, _ = nearest_points(COSTA, ponto)

    # Algarve: empurra para sul.
    if p_costa.y < 37.25 and p_costa.x > -8.95:
        return p_costa.y - deslocamento_graus, p_costa.x

    # Costa oeste: empurra para oeste.
    return p_costa.y, p_costa.x - deslocamento_graus


def ponto_em_mar_valido(lat, lon):
    """True se a coordenada pertence ao espaço marítimo português considerado."""
    return em_aguas_portugal(lat, lon)


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
    if hem in ('S', 'W'):
        dec = -dec
    return dec


def calcular_autonomia(combustivel_litros, consumo_litros_nm=CONSUMO_LITROS_NM_PADRAO):
    """
    Autonomia total (ida) e raio de ida-e-volta a partir do combustível disponível.

    Returns
    -------
    dict com alcance_total_nm, alcance_total_km, raio_ida_volta_nm, raio_ida_volta_km
    """
    if consumo_litros_nm <= 0 or combustivel_litros <= 0:
        return {
            'alcance_total_nm': 0.0,
            'alcance_total_km': 0.0,
            'raio_ida_volta_nm': 0.0,
            'raio_ida_volta_km': 0.0,
        }
    alcance_nm = combustivel_litros / consumo_litros_nm
    raio_nm = alcance_nm / 2.0
    return {
        'alcance_total_nm': alcance_nm,
        'alcance_total_km': alcance_nm * 1.852,
        'raio_ida_volta_nm': raio_nm,
        'raio_ida_volta_km': raio_nm * 1.852,
    }


def circulo_autonomia(lat, lon, raio_km, n_pontos=72):
    """Polígono (lon, lat) aproximando um círculo geodésico."""
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


def _importancia_de_severidade(severidade):
    return SEVERIDADE_IMPORTANCIA.get(str(severidade).strip(), 5.0)


def _marcar_acidente(row):
    vidas = pd.to_numeric(row.get('Lives lost Occurrence-Total', 0), errors='coerce') or 0
    if vidas > 0:
        return 1
    if str(row.get('Did the ship sink?', '')).strip().lower() == 'yes':
        return 1
    sev = str(row.get('Occurrence severity', '')).strip()
    if sev in ('Very serious', 'Serious'):
        return 1
    navio = str(row.get('Occurrence with ship(s)', '')).lower()
    if any(k in navio for k in ('collision', 'grounding', 'foundering', 'fire/explosion', 'flooding')):
        return 1
    return 0


def filtrar_incidentes_maritimos(df):
    """Mantém todas as ocorrências em território marítimo português.

    Remove apenas registos de portos, rios, águas interiores, estaleiros e
    coordenadas fora das janelas marítimas de Portugal continental, Açores e
    Madeira. Os pontos marítimos que aparecem ligeiramente em terra são
    corrigidos para visualização, não apagados.
    """
    df = df.copy()
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

    if 'Sea area of occurrence' in df.columns:
        df = df[~df['Sea area of occurrence'].isin(AREAS_TERRESTRES)]
        df = df[df['Sea area of occurrence'].isin(AREAS_MARITIMAS)]

    df['Teatro_Maritimo'] = [
        identificar_teatro_maritimo(lat, lon)
        for lat, lon in zip(df['Lat_original'], df['Lon_original'])
    ]
    df = df[df['Teatro_Maritimo'].notna()].copy()

    corrigidos = [
        corrigir_ponto_para_mar(lat, lon)
        for lat, lon in zip(df['Lat_original'], df['Lon_original'])
    ]
    if corrigidos:
        df['Lat'] = [p[0] for p in corrigidos]
        df['Lon'] = [p[1] for p in corrigidos]

    return df.reset_index(drop=True)


def carregar_incidentes_gama(caminho_csv):
    """
    Carrega o export GAMA, converte coordenadas DMS e filtra apenas o mar.
    Devolve DataFrame com Lat, Lon, Importancia, Acidente.
    """
    bruto = pd.read_csv(caminho_csv)
    return normalizar_incidentes_pontuais(bruto)


def normalizar_incidentes_pontuais(df_bruto):
    """Converte export GAMA ou CSV Lat/Lon num DataFrame marítimo padronizado."""
    mar = filtrar_incidentes_maritimos(df_bruto)
    if 'Importancia' not in mar.columns:
        if 'Occurrence severity' in mar.columns:
            mar['Importancia'] = mar['Occurrence severity'].apply(_importancia_de_severidade)
        else:
            mar['Importancia'] = 5.0
    if 'Acidente' not in mar.columns:
        mar['Acidente'] = mar.apply(_marcar_acidente, axis=1).astype(int)
    cols = ['Lat', 'Lon', 'Importancia', 'Acidente']
    for extra in ['Teatro_Maritimo', 'Lat_original', 'Lon_original']:
        if extra in mar.columns:
            cols.append(extra)
    return mar[cols].copy()


def exportar_incidentes_maritimos(caminho_entrada, caminho_saida):
    """Gera CSV limpo (apenas mar) a partir do export GAMA original."""
    limpo = carregar_incidentes_gama(caminho_entrada)
    bruto = pd.read_csv(caminho_entrada)
    mar = filtrar_incidentes_maritimos(bruto)
    mar.to_csv(caminho_saida, index=False)
    return len(bruto), len(limpo)


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
