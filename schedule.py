"""
=============================================================================
 SCHEDULE — standings reales, fixtures restantes, aplazados y jornadas
=============================================================================
Todo lo que responde "¿dónde estamos parados en la temporada?": tabla real
calculada de partidos.csv, qué partidos del calendario faltan por jugarse,
detección de aplazados, y a qué número de jornada pertenece cada partido.
"""
import pandas as pd

from config import PARTIDOS_CSV_PATH, normalizar_equipos


def tabla_real_actual(teams, liga="La Liga"):
    """Standings reales de la temporada en curso, calculados de partidos.csv."""
    p = pd.read_csv(PARTIDOS_CSV_PATH)
    p = p.dropna(subset=["goles_local", "goles_visitante"]).copy()
    p = normalizar_equipos(p, cols=("equipo_local", "equipo_visitante"))
    p = p[p.liga == liga]

    stats = {t: {"Pts": 0, "PJ": 0, "GF": 0, "GA": 0} for t in teams}
    for _, r in p.iterrows():
        gl, gv = int(r.goles_local), int(r.goles_visitante)
        for equipo, gf, ga in [(r.equipo_local, gl, gv), (r.equipo_visitante, gv, gl)]:
            if equipo not in stats:
                continue
            stats[equipo]["PJ"] += 1
            stats[equipo]["GF"] += gf
            stats[equipo]["GA"] += ga
            stats[equipo]["Pts"] += 3 if gf > ga else (1 if gf == ga else 0)

    tabla = pd.DataFrame(stats).T.reset_index().rename(columns={"index": "team"})
    tabla["GD"] = tabla.GF - tabla.GA
    return tabla.sort_values(["Pts", "GD"], ascending=False).reset_index(drop=True)


def fixtures_restantes(calendario, teams, liga="La Liga"):
    """Filtra un calendario a los partidos de esa liga que TODAVÍA no se jugaron."""
    p = pd.read_csv(PARTIDOS_CSV_PATH)
    p = normalizar_equipos(p, cols=("equipo_local", "equipo_visitante"))
    p = p[p.liga == liga]
    jugados = set(zip(p.equipo_local, p.equipo_visitante))

    cal = calendario.copy()
    ya_jugado = cal.apply(lambda r: (r.equipo_local, r.equipo_visitante) in jugados, axis=1)
    return cal[~ya_jugado].reset_index(drop=True)


def marcar_aplazados(calendario_completo, dias_umbral=12):
    """
    Agrega la columna `aplazado`: True si la fecha_programada de un partido
    quedó MÁS TARDE (no antes) que la fecha típica (mediana) de su misma
    jornada, por más de `dias_umbral` días. Recibe el calendario COMPLETO
    (con los ya jugados incluidos) para que la mediana de cada jornada se
    calcule sobre sus 10 partidos, no sobre lo que quede pendiente.
    """
    cal = calendario_completo.copy()
    cal["fecha_programada"] = pd.to_datetime(cal["fecha_programada"])
    cal["aplazado"] = False
    for _, grupo in cal.groupby("jornada"):
        fecha_tipica = grupo["fecha_programada"].median()
        dif_dias = (grupo["fecha_programada"] - fecha_tipica).dt.days  # con signo: solo tarde cuenta
        cal.loc[grupo.index[dif_dias > dias_umbral], "aplazado"] = True
    return cal


def fixtures_restantes_visibles(calendario_restante, calendario_completo, reference_date, dias_umbral=12):
    """
    Para VISTAS (grid de dificultad, jornadas) — no para el Monte Carlo: oculta
    un partido aplazado hasta que falten <= dias_umbral días para jugarse.
    Antes de eso, no cuenta como parte de su jornada (no la bloquea como
    "incompleta") ni aparece en la lista de próximos partidos.
    """
    if "fecha_programada" not in calendario_restante.columns:
        return calendario_restante
    marcados = marcar_aplazados(calendario_completo, dias_umbral)[
        ["jornada", "equipo_local", "equipo_visitante", "aplazado"]
    ]
    cal = calendario_restante.merge(marcados, on=["jornada", "equipo_local", "equipo_visitante"], how="left")
    cal["fecha_programada"] = pd.to_datetime(cal["fecha_programada"])
    dias_para_jugarse = (cal["fecha_programada"] - reference_date).dt.days
    ocultar = cal["aplazado"].fillna(False) & (dias_para_jugarse > dias_umbral)
    return cal[~ocultar].reset_index(drop=True)


def jornada_de_cada_partido(partidos_df, calendario, liga="La Liga"):
    """Devuelve partidos_df con una columna `jornada`, matcheando contra el calendario."""
    p = partidos_df[partidos_df.liga == liga].copy()
    cal_idx = calendario.set_index(["equipo_local", "equipo_visitante"])["jornada"]
    p["jornada"] = [
        cal_idx.get((r.equipo_local, r.equipo_visitante))
        for r in p.itertuples()
    ]
    return p.dropna(subset=["jornada"])


def detectar_jornadas(partidos_df, calendario, reference_date=None, liga="La Liga"):
    """
    (jornada_anterior, jornada_proxima):
      - anterior: la última jornada COMPLETA. Un partido aplazado (fecha muy
        alejada del resto de su jornada) no cuenta mientras falten más de 12
        días para jugarse -> su jornada puede darse por completa igual.
      - próxima:  la primera con partidos pendientes (visibles) posterior a esa
    """
    jugados = jornada_de_cada_partido(partidos_df, calendario, liga)
    if jugados.empty:
        return None, int(calendario.jornada.min())

    reference_date = reference_date or jugados.fecha.max() if "fecha" in jugados.columns else pd.Timestamp.now()
    cal_liga = calendario[calendario.liga == liga]
    restantes_full = fixtures_restantes(cal_liga, None, liga=liga)
    restantes_vis = fixtures_restantes_visibles(restantes_full, cal_liga, pd.Timestamp(reference_date))

    conteo_jugados = jugados.groupby("jornada").size()
    # total "exigible" por jornada = jugados + pendientes VISIBLES (los aplazados
    # ocultos no cuentan para exigir que la jornada esté completa)
    conteo_pendiente_visible = restantes_vis.groupby("jornada").size()
    conteo_exigido = conteo_jugados.add(conteo_pendiente_visible, fill_value=0)
    completas = conteo_jugados[conteo_jugados >= conteo_exigido.reindex(conteo_jugados.index).fillna(conteo_jugados)]
    jornada_anterior = int(completas.index.max()) if not completas.empty else int(jugados.jornada.max())

    jugados_clave = set(zip(jugados.equipo_local, jugados.equipo_visitante))
    pendientes = restantes_vis[
        ~restantes_vis.apply(lambda r: (r.equipo_local, r.equipo_visitante) in jugados_clave, axis=1)
    ]
    posteriores = pendientes[pendientes.jornada > jornada_anterior]
    jornada_proxima = int(posteriores.jornada.min()) if not posteriores.empty else None
    return jornada_anterior, jornada_proxima
