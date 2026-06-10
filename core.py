"""
Sistema de Apoio à Decisão para Patrulhamento Marítimo — núcleo.
Inclui: perfis de pesos, decaimento temporal e geometria de zonas (polígonos).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from math import radians, sin, cos, asin, sqrt
from datetime import datetime
from shapely.geometry import Point, Polygon

PERFIS = {
    'rotina':                          {'incidentes': 0.50, 'gravidade': 0.20, 'acidentes': 0.20, 'distancia': 0.10},
    'emergência':                      {'incidentes': 0.15, 'gravidade': 0.35, 'acidentes': 0.20, 'distancia': 0.30},
    'condições atmosféricas adversas': {'incidentes': 0.25, 'gravidade': 0.25, 'acidentes': 0.10, 'distancia': 0.40},
}

ZONAS_POLIGONOS = {
    1: Polygon([(-9.7, 38.4), (-9.3, 38.4), (-9.3, 39.0), (-9.7, 39.0)]),
    2: Polygon([(-9.0, 36.8), (-8.4, 36.8), (-8.4, 37.4), (-9.0, 37.4)]),
    3: Polygon([(-9.6, 39.2), (-9.0, 39.2), (-9.0, 39.8), (-9.6, 39.8)]),
    4: Polygon([(-8.2, 36.7), (-7.6, 36.7), (-7.6, 37.2), (-8.2, 37.2)]),
    5: Polygon([(-9.2, 39.9), (-8.7, 39.9), (-8.7, 40.5), (-9.2, 40.5)]),
    6: Polygon([(-14.0, 37.0), (-11.0, 37.0), (-11.0, 39.0), (-14.0, 39.0)]),
}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def distancia_a_poligono(pos_navio, poligono):
    lat, lon = pos_navio
    ponto = Point(lon, lat)
    if poligono.contains(ponto):
        return 0.0
    p_proximo = poligono.exterior.interpolate(poligono.exterior.project(ponto))
    return haversine_km(lat, lon, p_proximo.y, p_proximo.x)


def calcular_distancias(pos_navio, zonas=ZONAS_POLIGONOS):
    return {z: distancia_a_poligono(pos_navio, poly) for z, poly in zonas.items()}


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
