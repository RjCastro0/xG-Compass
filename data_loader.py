"""
=============================================================================
 DATA LOADER — carga y unificación del histórico, la temporada actual y el
 calendario multi-liga
=============================================================================
"""
import pandas as pd

from config import RAW_HIST_PATH, PARTIDOS_CSV_PATH, CALENDARIO_CSV_PATH, normalizar_equipos


def cargar_historico():
    """Histórico multi-liga 2000-2026 (all_time_df_actualizado.csv) al esquema base."""
    raw = pd.read_csv(RAW_HIST_PATH, low_memory=False)
    raw.columns = [c.strip().lstrip("\ufeff") for c in raw.columns]

    score = raw["Score"].astype(str).str.replace("–", "-", regex=False)
    partes = score.str.extract(r"(\d+)\D+(\d+)")
    raw["home_goals"] = pd.to_numeric(partes[0], errors="coerce")
    raw["away_goals"] = pd.to_numeric(partes[1], errors="coerce")
    raw = raw.dropna(subset=["home_goals", "away_goals"]).copy()
    raw[["home_goals", "away_goals"]] = raw[["home_goals", "away_goals"]].astype(int)

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"])
    raw = normalizar_equipos(raw, cols=("Home", "Away"))
    raw = normalizar_equipos(raw, cols=("League",))  # por si el nombre de liga trae alias

    return raw[["League", "Season", "Date", "Home", "Away", "home_goals", "away_goals"]]


def cargar_partidos_actuales():
    """partidos.csv (temporada 26-27, esquema propio) -> esquema del modelo."""
    p = pd.read_csv(PARTIDOS_CSV_PATH)
    p = p.dropna(subset=["goles_local", "goles_visitante"]).copy()

    out = pd.DataFrame({
        "League": p["liga"],
        "Season": "2026-2027",
        "Date": pd.to_datetime(p["fecha"], format="mixed", utc=True).dt.tz_localize(None),
        "Home": p["equipo_local"],
        "Away": p["equipo_visitante"],
        "home_goals": p["goles_local"].astype(int),
        "away_goals": p["goles_visitante"].astype(int),
    })
    return normalizar_equipos(out, cols=("Home", "Away"))


def construir_matches_df():
    """Concatena histórico + temporada actual, evitando duplicar partidos ya cargados."""
    hist = cargar_historico()
    actual = cargar_partidos_actuales()

    clave = lambda df: set(zip(df.Date.dt.date, df.Home, df.Away))
    ya_en_hist = clave(hist)
    actual_nuevo = actual[
        ~actual.apply(lambda r: (r.Date.date(), r.Home, r.Away) in ya_en_hist, axis=1)
    ]

    matches_df = pd.concat([hist, actual_nuevo], ignore_index=True)
    return matches_df.sort_values("Date").reset_index(drop=True)


def cargar_calendario(liga="La Liga"):
    """calendario_partidos.csv: fixture completo (multi-liga), filtrado a una liga."""
    cal = pd.read_csv(CALENDARIO_CSV_PATH)
    cal = normalizar_equipos(cal, cols=("equipo_local", "equipo_visitante"))
    return cal[cal.liga == liga].reset_index(drop=True)
