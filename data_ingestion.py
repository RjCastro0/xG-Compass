"""
=============================================================================
 DATA INGESTION — ingesta de CSV crudos (formato FBref) hacia el histórico y
 el calendario. Usado por agregar_datos.ipynb.
=============================================================================
"""
import os

import numpy as np
import pandas as pd

from config import (
    ALIAS, RAW_HIST_PATH, CALENDARIO_CSV_PATH, PARTIDOS_CSV_PATH,
    LEAGUE_NAME_MAP_HIST_TO_ACTUAL, normalizar_equipos, verificar_alias_conflictos,
)


def _parsear_fbref_crudo(fuente):
    """Lee un CSV crudo estilo FBref (Wk,Day,Date,Time,Home,Score,Away,...).
    `fuente` puede ser un path a archivo o el texto del CSV ya pegado.
    Tolera las filas separadoras en blanco entre jornadas."""
    import io
    if isinstance(fuente, str) and os.path.exists(fuente):
        raw = pd.read_csv(fuente)
    else:
        raw = pd.read_csv(io.StringIO(fuente))
    raw.columns = [c.strip() for c in raw.columns]
    raw = raw[raw["Wk"].notna()].reset_index(drop=True)
    return raw


def verificar_equipos_liga(equipos_normalizados, liga_partidos_csv):
    """Compara equipos ya normalizados contra los que partidos.csv ya reconoce
    para esa liga. Devuelve los que NO matchean (candidatos a alias faltante).
    Si partidos.csv no tiene esa liga todavía, no puede validar y devuelve []."""
    if not os.path.exists(PARTIDOS_CSV_PATH):
        return []
    actuales = normalizar_equipos(pd.read_csv(PARTIDOS_CSV_PATH), cols=("equipo_local", "equipo_visitante"))
    actuales = actuales[actuales.liga == liga_partidos_csv]
    equipos_actuales = set(actuales.equipo_local) | set(actuales.equipo_visitante)
    if not equipos_actuales:
        return []
    return sorted(set(equipos_normalizados) - equipos_actuales)


def agregar_temporada_historico(fuente, liga, season, hist_path=None):
    """
    Agrega una temporada completa (CSV crudo FBref) a all_time_df_actualizado.csv.

    fuente : path al CSV crudo, o el texto del CSV pegado directamente.
    liga   : nombre EXACTO que ya usa el histórico para esa liga (columna League).
             Usa uno de: 'La-Liga', 'Premier-League', 'Ligue-1', 'Seria-A',
             'Bundesliga', 'Eredivisie', 'Primeira-Liga'. Si es una liga nueva,
             usa el mismo nombre que quieras que tenga de ahora en más.
    season : '2024-2025', etc.

    Devuelve (df_resultado_completo, equipos_sin_matchear). Si la segunda lista
    no está vacía, imprime una advertencia: hay que agregar esos alias a ALIAS
    (en config.py) para que el equipo matchee con partidos.csv.
    """
    verificar_alias_conflictos()
    hist_path = hist_path or RAW_HIST_PATH
    raw = _parsear_fbref_crudo(fuente)

    home_norm = raw["Home"].replace(ALIAS)
    away_norm = raw["Away"].replace(ALIAS)
    equipos_normalizados = sorted(set(home_norm) | set(away_norm))
    liga_actual = LEAGUE_NAME_MAP_HIST_TO_ACTUAL.get(liga, liga)
    faltantes = verificar_equipos_liga(equipos_normalizados, liga_actual)
    if faltantes:
        print(f"⚠️  {len(faltantes)} equipo(s) no matchean con partidos.csv para '{liga_actual}' "
              "— probablemente falta un alias en ALIAS:")
        for f in faltantes:
            print("   -", f)

    nuevos = pd.DataFrame({
        "League": liga, "Season": season, "Wk": raw["Wk"], "Day": raw.get("Day"),
        "Date": raw["Date"], "Time": raw.get("Time"), "Home": home_norm,
        "Score": raw["Score"], "Away": away_norm,
        "Attendance": raw.get("Attendance"), "Venue": raw.get("Venue"),
        "Referee": raw.get("Referee"), "Match Report": raw.get("Match Report"),
        "Notes": raw.get("Notes"), "xG": np.nan, "xG.1": np.nan, "Round": np.nan,
    })

    hist = pd.read_csv(hist_path, low_memory=False)
    hist.columns = [c.strip().lstrip("\ufeff") for c in hist.columns]
    existentes = set(zip(hist.League, hist.Date.astype(str), hist.Home, hist.Away))
    dup_mask = nuevos.apply(
        lambda r: (r.League, str(r.Date), r.Home, r.Away) in existentes, axis=1
    )
    print(f"Partidos en el CSV: {len(nuevos)} | ya existían (se omiten): {int(dup_mask.sum())} "
          f"| nuevos a agregar: {int((~dup_mask).sum())}")
    nuevos = nuevos[~dup_mask]

    combinado = pd.concat([hist, nuevos], ignore_index=True)
    combinado.to_csv(hist_path, index=False)
    print(f"Guardado: {hist_path} ({len(hist)} -> {len(combinado)} filas)")
    return combinado, faltantes


def agregar_liga_calendario(fuente, liga, calendario_path=None):
    """
    Agrega o reemplaza el calendario de una liga en calendario_partidos.csv,
    a partir de un CSV crudo FBref (usa Wk, Home, Away, Date; ignora el resto,
    incluidos los resultados si los trae).

    fuente : path al CSV crudo, o el texto del CSV pegado directamente.
    liga   : nombre tal como lo usa partidos.csv (ej. 'Serie A', 'Premier League').

    Devuelve (df_calendario_completo, equipos_sin_matchear).
    """
    verificar_alias_conflictos()
    calendario_path = calendario_path or CALENDARIO_CSV_PATH
    raw = _parsear_fbref_crudo(fuente)

    home_norm = raw["Home"].replace(ALIAS)
    away_norm = raw["Away"].replace(ALIAS)
    equipos_normalizados = sorted(set(home_norm) | set(away_norm))
    faltantes = verificar_equipos_liga(equipos_normalizados, liga)
    if faltantes:
        print(f"⚠️  {len(faltantes)} equipo(s) no matchean con partidos.csv para '{liga}' "
              "— probablemente falta un alias en ALIAS:")
        for f in faltantes:
            print("   -", f)

    nuevo_cal = pd.DataFrame({
        "liga": liga, "jornada": raw["Wk"], "equipo_local": home_norm,
        "equipo_visitante": away_norm, "fecha_programada": raw["Date"],
    })

    if os.path.exists(calendario_path):
        actual = pd.read_csv(calendario_path)
        actual = actual[actual.liga != liga]  # reemplaza la liga si ya estaba cargada
    else:
        actual = pd.DataFrame(columns=nuevo_cal.columns)

    combinado = pd.concat([actual, nuevo_cal], ignore_index=True)
    combinado.to_csv(calendario_path, index=False)
    print(f"Guardado: {calendario_path} ({len(actual)} -> {len(combinado)} filas, "
          f"{len(nuevo_cal)} de '{liga}')")
    return combinado, faltantes
