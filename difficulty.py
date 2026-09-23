"""
=============================================================================
 DIFFICULTY — score de dificultad de fixture (1-5, estilo FPL/FFH)
=============================================================================
"""
import numpy as np
import pandas as pd


def _mi_forma_actual(equipo, es_local, partidos_temporada, half_life_partidos=7.0):
    """
    Compuesto de forma ofensiva actual: xG generado + goles realmente
    metidos, ponderados por su correlación medida con los goles del
    siguiente partido (xG r=+0.530, goles r=+0.455 sobre 72-78 obs).
    NOTA: xG y goles correlacionan 0.71 entre sí (r=-0.431 solo con xG vs
    r=-0.382 con este compuesto, en la validación de dificultad completa)
    — incluir goles resta algo de precisión predictiva por esa redundancia,
    pero se mantiene a pedido para que la fórmula refleje también el
    resultado real, no solo la calidad de las ocasiones.
    """
    W_XG, W_GOLES = 0.530 / (0.530 + 0.455), 0.455 / (0.530 + 0.455)

    liga_xg = partidos_temporada["xg_local"].mean()
    liga_goles = pd.concat([partidos_temporada.goles_local,
                             partidos_temporada.goles_visitante]).mean()

    mi_xg = _xg_favor_prom(equipo, es_local, partidos_temporada, half_life_partidos) or liga_xg
    mi_goles = _metrica_prom(equipo, es_local, partidos_temporada, "goles", half_life_partidos) or liga_goles

    return W_XG * (mi_xg / liga_xg) + W_GOLES * (mi_goles / liga_goles)


def _metrica_prom(equipo, es_local, partidos_temporada, columna, half_life_partidos=7.0):
    """Promedio ponderado por recencia (decae por PARTIDOS) de cualquier métrica _local/_visitante."""
    col_eq = "equipo_local" if es_local else "equipo_visitante"
    col_val = f"{columna}_{'local' if es_local else 'visitante'}"
    v = partidos_temporada[partidos_temporada[col_eq] == equipo][col_val].values
    if len(v) == 0:
        return None
    n = len(v)
    pesos = 0.5 ** (np.arange(n - 1, -1, -1) / half_life_partidos)
    pesos /= pesos.sum()
    return float(np.dot(pesos, v))


def _xg_favor_prom(equipo, es_local, partidos_temporada, half_life_partidos=7.0):
    """xG generado por `equipo`, ponderado por recencia (decae por PARTIDOS, no días)."""
    col_eq = "equipo_local" if es_local else "equipo_visitante"
    col_xg = "xg_local" if es_local else "xg_visitante"
    v = partidos_temporada[partidos_temporada[col_eq] == equipo][col_xg].values
    if len(v) == 0:
        return None
    n = len(v)
    pesos = 0.5 ** (np.arange(n - 1, -1, -1) / half_life_partidos)
    pesos /= pesos.sum()
    return float(np.dot(pesos, v))


def _xg_concedido_prom(equipo, es_local, partidos_temporada, half_life_partidos=7.0):
    """xG concedido por `equipo` (= xG que generó su oponente en esos partidos)."""
    col_eq = "equipo_local" if es_local else "equipo_visitante"
    col_xg_opp = "xg_visitante" if es_local else "xg_local"
    v = partidos_temporada[partidos_temporada[col_eq] == equipo][col_xg_opp].values
    if len(v) == 0:
        return None
    n = len(v)
    pesos = 0.5 ** (np.arange(n - 1, -1, -1) / half_life_partidos)
    pesos /= pesos.sum()
    return float(np.dot(pesos, v))


def score_dificultad_bruto(mi_equipo, rival, es_local, hist_model, partidos_temporada,
                            reference_date, liga=None):
    """
    Dificultad para MI_EQUIPO de enfrentar a RIVAL. Mayor score = más difícil.

    Combina el rating histórico (estable, largo plazo) del rival con un
    ajuste por forma actual: compuesto xG+goles propio, y xG concedido del
    rival esta temporada.

    Validado contra los goles reales del siguiente partido de cada equipo:
      - solo rating histórico:        r=+0.17
      - + ajuste con SOLO xG:         r=-0.43
      - + ajuste con xG+goles:        r=-0.38  (xG y goles correlacionan 0.71
        entre sí, así que agregar goles resta algo de precisión — se
        mantiene igual porque refleja el resultado real, no solo xG)
    """
    if liga is not None:
        partidos_temporada = partidos_temporada[partidos_temporada.liga == liga]

    ha, hd, aa, ad = hist_model.get_ratings(rival, reference_date)
    rival_ataque_hist = aa if es_local else ha

    liga_xg = partidos_temporada["xg_local"].mean()
    mi_forma = _mi_forma_actual(mi_equipo, es_local, partidos_temporada)
    rival_xgc = _xg_concedido_prom(rival, not es_local, partidos_temporada) or liga_xg

    ratio_forma = mi_forma * (rival_xgc / liga_xg)
    return rival_ataque_hist - ratio_forma


def tabla_dificultad(hist_model, partidos_df, calendario, jornadas, reference_date, liga="La Liga"):
    """
    Fixture ticker: una fila por equipo, una columna por jornada, con la
    dificultad normalizada a 1-5 (percentiles sobre TODOS los fixtures
    de esa misma jornada, para que "5" siempre signifique "de lo más duro
    de esa fecha", no un score absoluto que no compara entre jornadas).
    """
    partidos_temporada = partidos_df[partidos_df.liga == liga]
    filas = []
    for jornada in jornadas:
        partidos_jornada = calendario[calendario.jornada == jornada]
        scores_brutos = []
        registros = []
        for _, row in partidos_jornada.iterrows():
            for equipo, rival, es_local in [
                (row.equipo_local, row.equipo_visitante, True),
                (row.equipo_visitante, row.equipo_local, False),
            ]:
                s = score_dificultad_bruto(equipo, rival, es_local, hist_model,
                                            partidos_temporada, reference_date)
                scores_brutos.append(s)
                registros.append((jornada, equipo, rival, es_local, s))

        # normalizar a 1-5 dentro de la jornada (5 = más difícil)
        rangos = pd.qcut(scores_brutos, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
        for (jr, equipo, rival, es_local, _), dificultad in zip(registros, rangos):
            filas.append({
                "jornada": jr, "equipo": equipo, "rival": rival,
                "H_A": "H" if es_local else "A",
                "dificultad": int(dificultad),
            })
    return pd.DataFrame(filas)
