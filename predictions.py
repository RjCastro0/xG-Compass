"""
=============================================================================
 PREDICTIONS — ratings de equipo, predicción de partido puntual, y
 análisis/predicciones por jornada
=============================================================================
"""
import numpy as np
import pandas as pd

from goal_models import _poisson_pmf, outcome_probabilities
from schedule import jornada_de_cada_partido


def construir_rating_table(goal_model, teams, reference_date):
    """Ratings de todos los equipos de la liga, a una fecha de referencia."""
    filas = []
    for team in teams:
        ha, hd, aa, ad = goal_model.get_ratings(team, reference_date)
        filas.append({
            "team": team,
            "home_attack": ha, "home_defense": hd,
            "away_attack": aa, "away_defense": ad,
        })
    return pd.DataFrame(filas).sort_values("home_attack", ascending=False).reset_index(drop=True)


def predecir_partido(goal_model, home_team, away_team, reference_date, max_goals=8):
    """Devuelve lambdas, 1X2 y el marcador más probable para un partido."""
    if hasattr(goal_model, "score_matrix"):
        matrix, lh, la = goal_model.score_matrix(home_team, away_team, reference_date, max_goals)
        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()
    else:
        lh, la = goal_model.expected_goals(home_team, away_team, reference_date)
        p_home, p_draw, p_away = outcome_probabilities(lh, la)
        home_probs = np.array([_poisson_pmf(k, lh) for k in range(max_goals)])
        away_probs = np.array([_poisson_pmf(k, la) for k in range(max_goals)])
        matrix = np.outer(home_probs, away_probs)

    h_score, a_score = np.unravel_index(np.argmax(matrix), matrix.shape)

    return {
        "home_team": home_team, "away_team": away_team,
        "lambda_home": lh, "lambda_away": la,
        "p_home": p_home, "p_draw": p_draw, "p_away": p_away,
        "xscore": f"{h_score}-{a_score}",
    }


def analisis_jornada_jugada(partidos_df, calendario, jornada, liga="La Liga"):
    """
    Resultados + métricas de calidad de una jornada ya jugada.
    Por equipo:
      xGD               = xG propio - xG concedido (dominio del partido)
      sobre_rendimiento = goles - xG (positivo = finalizó mejor de lo esperado;
                          valores altos sostenidos suelen revertirse a la media)
    """
    jugados = jornada_de_cada_partido(partidos_df, calendario, liga)
    jugados = jugados[jugados.jornada == jornada]

    filas = []
    for r in jugados.itertuples():
        filas.append({
            "local": r.equipo_local, "visitante": r.equipo_visitante,
            "goles_local": int(r.goles_local), "goles_visitante": int(r.goles_visitante),
            "xg_local": r.xg_local, "xg_visitante": r.xg_visitante,
            "xgd_local": r.xg_local - r.xg_visitante,
            "xgd_visitante": r.xg_visitante - r.xg_local,
            "sobre_local": r.goles_local - r.xg_local,
            "sobre_visitante": r.goles_visitante - r.xg_visitante,
        })
    return pd.DataFrame(filas)


def analisis_ultimos_partidos(partidos_df, liga="La Liga", n=10):
    """
    Igual que analisis_jornada_jugada, pero sin depender del calendario:
    toma los últimos `n` partidos jugados de partidos.csv por fecha, ordenados
    del más reciente al más viejo. Sirve de respaldo cuando el calendario de
    una liga no cubre todavía las jornadas ya jugadas (no se puede saber a
    qué jornada pertenece cada partido, pero el resultado ya está en
    partidos.csv de todas formas).
    """
    p = partidos_df[partidos_df.liga == liga].dropna(subset=["goles_local", "goles_visitante"]).copy()
    p["fecha"] = pd.to_datetime(p["fecha"])
    p = p.sort_values("fecha", ascending=False).head(n)

    filas = []
    for r in p.itertuples():
        filas.append({
            "fecha": r.fecha.date(),
            "local": r.equipo_local, "visitante": r.equipo_visitante,
            "goles_local": int(r.goles_local), "goles_visitante": int(r.goles_visitante),
            "xg_local": r.xg_local, "xg_visitante": r.xg_visitante,
            "xgd_local": r.xg_local - r.xg_visitante,
            "xgd_visitante": r.xg_visitante - r.xg_local,
            "sobre_local": r.goles_local - r.xg_local,
            "sobre_visitante": r.goles_visitante - r.xg_visitante,
        })
    return pd.DataFrame(filas)


def predicciones_jornada(goal_model, calendario, partidos_df, jornada,
                          reference_date, liga="La Liga"):
    """Predicción de todos los partidos PENDIENTES de una jornada."""
    jugados = jornada_de_cada_partido(partidos_df, calendario, liga)
    jugados_clave = set(zip(jugados.equipo_local, jugados.equipo_visitante))

    de_la_jornada = calendario[calendario.jornada == jornada]
    pendientes = de_la_jornada[
        ~de_la_jornada.apply(lambda r: (r.equipo_local, r.equipo_visitante) in jugados_clave, axis=1)
    ]

    filas = []
    for r in pendientes.itertuples():
        pred = predecir_partido(goal_model, r.equipo_local, r.equipo_visitante, reference_date)
        filas.append({
            "local": r.equipo_local, "visitante": r.equipo_visitante,
            "xscore": pred["xscore"],
            "xg_local": pred["lambda_home"], "xg_visitante": pred["lambda_away"],
            "p_local": pred["p_home"], "p_empate": pred["p_draw"], "p_visitante": pred["p_away"],
        })
    return pd.DataFrame(filas)
