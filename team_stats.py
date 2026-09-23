"""
=============================================================================
 TEAM STATS — estadísticas de equipo para el tab Stats (Stats/Ataque/Defensa)
=============================================================================
"""
import numpy as np
import pandas as pd


def calcular_stats_radar(partidos_df, teams, cols, liga="La Liga"):
    """
    Promedio por partido de las columnas pedidas (local+visitante combinados),
    para cualquier radar comparativo de hasta 3 equipos. `cols` son los nombres
    base sin sufijo (ej. "tackles" -> usa tackles_local/tackles_visitante).
    """
    p = partidos_df[partidos_df.liga == liga].dropna(subset=["goles_local", "goles_visitante"])

    L = pd.DataFrame({"equipo": p.equipo_local, **{c: p[f"{c}_local"] for c in cols}})
    V = pd.DataFrame({"equipo": p.equipo_visitante, **{c: p[f"{c}_visitante"] for c in cols}})
    larga = pd.concat([L, V], ignore_index=True)

    filas = []
    for team in teams:
        obs = larga[larga.equipo == team]
        if obs.empty:
            continue
        fila = {"equipo": team}
        for c in cols:
            fila[c] = obs[c].mean()
        filas.append(fila)
    return pd.DataFrame(filas)


def calcular_forma_reciente(partidos_df, teams, n=5, liga="La Liga"):
    """
    Por equipo, agregados de sus últimos `n` partidos jugados (cronológico):
    puntos, xGD (xG-xGA), xG, xGA, goles, GA — todo SUMADO sobre esos n partidos.
    """
    p = partidos_df[partidos_df.liga == liga].dropna(subset=["goles_local", "goles_visitante"]).copy()
    p["fecha"] = pd.to_datetime(p["fecha"])

    L = pd.DataFrame({
        "fecha": p.fecha, "equipo": p.equipo_local,
        "goles": p.goles_local, "goles_c": p.goles_visitante,
        "xg": p.xg_local, "xga": p.xg_visitante,
        "pts": np.where(p.goles_local > p.goles_visitante, 3,
                np.where(p.goles_local == p.goles_visitante, 1, 0)),
    })
    V = pd.DataFrame({
        "fecha": p.fecha, "equipo": p.equipo_visitante,
        "goles": p.goles_visitante, "goles_c": p.goles_local,
        "xg": p.xg_visitante, "xga": p.xg_local,
        "pts": np.where(p.goles_visitante > p.goles_local, 3,
                np.where(p.goles_local == p.goles_visitante, 1, 0)),
    })
    larga = pd.concat([L, V], ignore_index=True).sort_values("fecha")

    filas = []
    for team in teams:
        obs = larga[larga.equipo == team].tail(n)
        if obs.empty:
            continue
        filas.append({
            "equipo": team, "n_partidos": len(obs),
            "puntos_last5": obs.pts.sum(),
            "xgd_last5": (obs.xg - obs.xga).sum(),
            "xg_last5": obs.xg.sum(), "xga_last5": obs.xga.sum(),
            "xga_prom_last5": obs.xga.mean(),
            "goles_last5": obs.goles.sum(), "ga_last5": obs.goles_c.sum(),
        })
    return pd.DataFrame(filas)


def calcular_rates_modelo(goal_model, teams, reference_date):
    """Rate de ataque/defensa por equipo, promedio de su faceta local+visitante del modelo vigente."""
    filas = []
    for team in teams:
        ha, hd, aa, ad = goal_model.get_ratings(team, reference_date)
        filas.append({"equipo": team, "rate_ataque": (ha + aa) / 2, "rate_defensa": (hd + ad) / 2})
    return pd.DataFrame(filas)


def calcular_stats_equipos(partidos_df, teams, liga="La Liga"):
    """
    Una fila por equipo con las métricas del tab Stats/Ataque/Defensa.
    Todas las métricas "/partido" son promedio sobre los partidos jugados
    esta temporada (local + visitante combinados).
    """
    p = partidos_df[partidos_df.liga == liga].dropna(subset=["goles_local", "goles_visitante"])

    L = pd.DataFrame({
        "equipo": p.equipo_local, "goles": p.goles_local, "goles_c": p.goles_visitante,
        "xg": p.xg_local, "xga": p.xg_visitante, "xgot": p.xgot_local,
        "tiros": p.tiros_local, "tiros_c": p.tiros_visitante,
        "tiros_puerta": p.tiros_puerta_local,
        "faltas": p.faltas_local, "amarillas": p.amarillas_local, "rojas": p.rojas_local,
        "posesiones_ut": p.posesiones_ganadas_ultimo_tercio_local,
        "resultado": np.where(p.goles_local > p.goles_visitante, "W",
                      np.where(p.goles_local == p.goles_visitante, "D", "L")),
    })
    V = pd.DataFrame({
        "equipo": p.equipo_visitante, "goles": p.goles_visitante, "goles_c": p.goles_local,
        "xg": p.xg_visitante, "xga": p.xg_local, "xgot": p.xgot_visitante,
        "tiros": p.tiros_visitante, "tiros_c": p.tiros_local,
        "tiros_puerta": p.tiros_puerta_visitante,
        "faltas": p.faltas_visitante, "amarillas": p.amarillas_visitante, "rojas": p.rojas_visitante,
        "posesiones_ut": p.posesiones_ganadas_ultimo_tercio_visitante,
        "resultado": np.where(p.goles_visitante > p.goles_local, "W",
                      np.where(p.goles_local == p.goles_visitante, "D", "L")),
    })
    larga = pd.concat([L, V], ignore_index=True)

    filas = []
    for team in teams:
        obs = larga[larga.equipo == team]
        pj = len(obs)
        if pj == 0:
            continue
        filas.append({
            "equipo": team, "PJ": pj,
            "xG/partido": obs.xg.mean(), "xGA/partido": obs.xga.mean(),
            "Goles": obs.goles.sum(), "GA": obs.goles_c.sum(),
            "Win%": (obs.resultado == "W").mean() * 100,
            "Clean sheets": (obs.goles_c == 0).sum(),
            "%Tiros anotados": (obs.goles.sum() / obs.tiros.sum() * 100) if obs.tiros.sum() else 0,
            "%Tiros a puerta": (obs.tiros_puerta.sum() / obs.tiros.sum() * 100) if obs.tiros.sum() else 0,
            "Faltas": obs.faltas.sum(),
            "Amarillas/partido": obs.amarillas.mean(),
            "Rojas": obs.rojas.sum(),
            "xGOT/partido": obs.xgot.mean(),
            "Tiros/partido": obs.tiros.mean(),
            "Tiros permitidos/partido": obs.tiros_c.mean(),
            "Posesiones UT/partido": obs.posesiones_ut.mean(),
        })
    return pd.DataFrame(filas)
