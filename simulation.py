"""
=============================================================================
 SIMULATION — Monte Carlo sobre el calendario real restante
=============================================================================
"""
import numpy as np
import pandas as pd

from config import N_SIMULATIONS, PARTIDOS_CSV_PATH, normalizar_equipos
from data_loader import cargar_calendario
from goal_models import outcome_probabilities, puntos_esperados


def simular_temporada(goal_model, calendario_restante, tabla_real, teams,
                       reference_date, n_simulations=N_SIMULATIONS, seed=0):
    """
    Simula n_simulations finales de temporada arrancando desde los puntos
    reales YA obtenidos, y jugando el calendario REAL restante (no uno
    sintético) partido a partido vía Poisson.
    """
    rng = np.random.default_rng(seed)
    team_index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    pts_base = tabla_real.set_index("team")["Pts"].reindex(teams).fillna(0).values
    gf_base = tabla_real.set_index("team")["GF"].reindex(teams).fillna(0).values
    ga_base = tabla_real.set_index("team")["GA"].reindex(teams).fillna(0).values

    usa_tau = hasattr(goal_model, "score_matrix")
    max_goals = 10

    # pre-computar, para cada partido restante, la distribución completa de
    # marcador (con tau si el modelo lo soporta) -> no cambia entre simulaciones
    partidos_pre = []
    for _, row in calendario_restante.iterrows():
        ih, ia = team_index[row.equipo_local], team_index[row.equipo_visitante]
        if usa_tau:
            matrix, lh, la = goal_model.score_matrix(row.equipo_local, row.equipo_visitante,
                                                       reference_date, max_goals)
            partidos_pre.append((ih, ia, matrix.ravel()))
        else:
            lh, la = goal_model.expected_goals(row.equipo_local, row.equipo_visitante, reference_date)
            partidos_pre.append((ih, ia, lh, la))

    points = np.tile(pts_base, (n_simulations, 1))
    goals_for = np.tile(gf_base, (n_simulations, 1))
    goals_against = np.tile(ga_base, (n_simulations, 1))

    for item in partidos_pre:
        if usa_tau:
            ih, ia, probs_flat = item
            # muestrear un indice 0..max_goals^2-1 de la distribucion conjunta
            # (con tau) y decodificarlo a (goles_local, goles_visitante)
            idx = rng.choice(len(probs_flat), size=n_simulations, p=probs_flat)
            hg, ag = np.divmod(idx, max_goals)
        else:
            ih, ia, lh, la = item
            hg = rng.poisson(lh, size=n_simulations)
            ag = rng.poisson(la, size=n_simulations)

        goals_for[:, ih] += hg; goals_against[:, ih] += ag
        goals_for[:, ia] += ag; goals_against[:, ia] += hg
        home_win = hg > ag
        away_win = hg < ag
        draw = ~home_win & ~away_win
        points[:, ih] += np.where(home_win, 3, np.where(draw, 1, 0))
        points[:, ia] += np.where(away_win, 3, np.where(draw, 1, 0))

    final_rank = (-points).argsort(axis=1).argsort(axis=1) + 1
    return points, goals_for, goals_against, final_rank, team_index


def tabla_proyectada(points, goals_for, goals_against, final_rank, team_index, teams):
    """Standings proyectados: promedio sobre todas las temporadas simuladas."""
    filas = []
    for team in teams:
        i = team_index[team]
        ranks = final_rank[:, i]
        filas.append({
            "team": team,
            "exp_Pts": int(round(points[:, i].mean())),
            "exp_GD": int(round((goals_for[:, i] - goals_against[:, i]).mean())),
            "title_pct": round((ranks == 1).mean() * 100, 1),
            "top4_pct": round((ranks <= 4).mean() * 100, 1),
            "relegation_pct": round((ranks >= 18).mean() * 100, 1),
            "median_finish": int(np.median(ranks)),
        })
    tabla = pd.DataFrame(filas).sort_values(["exp_Pts", "exp_GD"], ascending=False)
    return tabla.reset_index(drop=True)


def progreso_puntos_combinado(goal_model, teams, reference_date, liga="La Liga"):
    """
    Para el line chart tipo 'progreso de puntos por jornada':
      - Jornadas ya jugadas: puntos reales acumulados (escalones 0/1/3).
      - Jornadas restantes: puntos ESPERADOS acumulados (valor esperado
        3*p_home + 1*p_draw, no una simulación puntual) -> curva suave.
    Se devuelve un DataFrame ancho: filas = jornada, columnas = equipo.
    """
    p = pd.read_csv(PARTIDOS_CSV_PATH)
    p = p.dropna(subset=["goles_local", "goles_visitante"]).copy()
    p = normalizar_equipos(p, cols=("equipo_local", "equipo_visitante"))
    p = p[p.liga == liga]

    cal = cargar_calendario(liga=liga)
    max_jornada = cal.jornada.max()

    # --- puntos reales por jornada jugada ---
    # nota: partidos.csv no trae "jornada"; se infiere la jornada de cada
    # partido real haciendo match contra el calendario (local+visitante)
    cal_idx = cal.set_index(["equipo_local", "equipo_visitante"])["jornada"]
    reales = []
    for _, r in p.iterrows():
        jr = cal_idx.get((r.equipo_local, r.equipo_visitante))
        if jr is None:
            continue
        gl, gv = int(r.goles_local), int(r.goles_visitante)
        reales.append((jr, r.equipo_local, 3 if gl > gv else (1 if gl == gv else 0)))
        reales.append((jr, r.equipo_visitante, 3 if gv > gl else (1 if gl == gv else 0)))
    df_real = pd.DataFrame(reales, columns=["jornada", "team", "pts"])

    tabla_ancha = pd.DataFrame(index=range(1, max_jornada + 1), columns=teams, dtype=float).fillna(0.0)
    for (jr, team), grupo in df_real.groupby(["jornada", "team"]):
        if team in tabla_ancha.columns:
            tabla_ancha.loc[jr, team] = grupo.pts.sum()

    ultima_jornada_jugada = df_real.jornada.max() if not df_real.empty else 0

    # --- puntos esperados por jornada restante ---
    restantes = cal[cal.jornada > ultima_jornada_jugada]
    for _, row in restantes.iterrows():
        lh, la = goal_model.expected_goals(row.equipo_local, row.equipo_visitante, reference_date)
        p_home, p_draw, p_away = outcome_probabilities(lh, la)
        xpts_local, xpts_visit = puntos_esperados(p_home, p_draw, p_away)
        if row.equipo_local in tabla_ancha.columns:
            tabla_ancha.loc[row.jornada, row.equipo_local] = xpts_local
        if row.equipo_visitante in tabla_ancha.columns:
            tabla_ancha.loc[row.jornada, row.equipo_visitante] = xpts_visit

    return tabla_ancha.cumsum()
