"""
=============================================================================
 FOOTBALL ANALYTICS 2026-2027 — STREAMLIT DASHBOARD (main entry point)
=============================================================================
Run with:  streamlit run App.py

Tabs (each in its own module):
  1) tab_difficulty  — team x matchday grid, green (easy) -> red (hard)
  2) tab_matchdays   — previous matchday played | upcoming matchday
  3) tab_standings   — current real table vs. projected table (Monte Carlo)
  4) tab_stats       — Stats / Attack / Defense

The model is fit ONCE per league and cached here; every tab module receives
the already-computed state as plain function arguments, so they're always
in sync with each other and never re-fit the model themselves.
"""
import pandas as pd
import streamlit as st

from config import PARTIDOS_CSV_PATH, normalizar_equipos
from data_loader import construir_matches_df, cargar_calendario
from goal_models import TimeDecayGoalModel, HybridGoalModel
from config import HALF_LIFE_DAYS, SHRINKAGE_STRENGTH
from predictions import construir_rating_table
from schedule import tabla_real_actual, fixtures_restantes, fixtures_restantes_visibles, jornada_de_cada_partido
from simulation import simular_temporada, tabla_proyectada

import tab_difficulty
import tab_matchdays
import tab_standings
import tab_stats

st.set_page_config(page_title="Football Analytics", layout="wide")


# =============================================================================
# SECTION 1 — CACHED LOAD AND FIT
# =============================================================================
# @st.cache_resource keeps the model object across Streamlit re-runs.
# It only recomputes when the CSVs change (cleared with the sidebar button).
#
# Two models with different roles, each validated separately:
#   - prior_model: pure historical (goals only) -> input for fixture difficulty
#   - goal_model:  hybrid (current-season composite) + Dixon-Coles correction
#                  -> match predictions and season Monte Carlo
#                  (64.1% walk-forward accuracy, vs 53.8% for the goals-only prior)
LIGAS_DISPONIBLES = ["La Liga", "Premier League", "Serie A", "Bundesliga", "Ligue 1"]

st.title("⚽ Football Analytics")
liga_seleccionada = st.selectbox("League", LIGAS_DISPONIBLES, key="liga_global")
st.divider()


@st.cache_resource(show_spinner="Fitting model...")
def cargar_modelo(liga):
    matches_df = construir_matches_df()
    reference_date = matches_df["Date"].max()
    partidos_temporada = normalizar_equipos(pd.read_csv(PARTIDOS_CSV_PATH),
                                             cols=("equipo_local", "equipo_visitante"))
    p_liga = partidos_temporada[partidos_temporada.liga == liga]
    equipos = sorted(set(p_liga.equipo_local) | set(p_liga.equipo_visitante))

    prior_model = TimeDecayGoalModel(HALF_LIFE_DAYS, SHRINKAGE_STRENGTH).fit(matches_df)
    goal_model = HybridGoalModel().fit(matches_df, partidos_temporada, liga=liga)
    goal_model.fit_rho(partidos_temporada, liga=liga)

    rating_table = construir_rating_table(goal_model, equipos, reference_date)
    return (goal_model, prior_model, rating_table, reference_date,
            len(matches_df), partidos_temporada, equipos)


@st.cache_data(show_spinner="Simulating season...")
def cargar_proyeccion(_goal_model, reference_date, liga, equipos):
    """The underscore in _goal_model keeps Streamlit from trying to hash the model."""
    calendario = cargar_calendario(liga=liga)
    real = tabla_real_actual(equipos, liga=liga)
    restantes = fixtures_restantes(calendario, equipos, liga=liga)
    points, gf, ga, ranks, idx = simular_temporada(
        _goal_model, restantes, real, equipos, reference_date
    )
    proyectada = tabla_proyectada(points, gf, ga, ranks, idx, equipos)
    return real, proyectada, calendario, restantes


goal_model, prior_model, rating_table, reference_date, n_matches, partidos_temporada, equipos_liga = (
    cargar_modelo(liga_seleccionada)
)
real, proyectada, calendario, restantes = cargar_proyeccion(
    goal_model, reference_date, liga_seleccionada, equipos_liga
)
# restantes_visibles: for the difficulty grid and matchdays -> hides a postponed
# match until <=12 days are left before it's played. The Monte Carlo (above)
# still uses the full, unfiltered `restantes` so the simulation doesn't lose matches.
restantes_visibles = fixtures_restantes_visibles(restantes, calendario, reference_date)

CALENDARIO_INCOMPLETO = jornada_de_cada_partido(partidos_temporada, calendario, liga=liga_seleccionada).empty \
    and not partidos_temporada[partidos_temporada.liga == liga_seleccionada].dropna(subset=["goles_local"]).empty
if CALENDARIO_INCOMPLETO:
    st.caption(
        f"⚠️ The **{liga_seleccionada}** calendar doesn't cover the first matchdays yet "
        "(already played), so the exact matchday number can't be determined for those "
        "matches — in the **Matchdays** tab, 'Previous matchday' shows the latest results "
        "by date instead. Difficulty, upcoming matchdays and Standings work as usual."
    )


# =============================================================================
# SECTION 2 — SIDEBAR
# =============================================================================
with st.sidebar:
    st.header(f"{liga_seleccionada} 2026-2027")
    st.caption(f"{n_matches:,} matches in the model")
    st.caption(f"Data through {reference_date.date()}")
    if st.button("Reload data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

tab_fdr, tab_pred, tab_tabla, tab_stat = st.tabs(
    ["Fixture Difficulty", "Matchdays", "Standings", "Stats"]
)

with tab_fdr:
    tab_difficulty.render(
        prior_model, partidos_temporada, restantes_visibles, reference_date,
        liga_seleccionada, equipos_liga,
    )

with tab_pred:
    tab_matchdays.render(
        partidos_temporada, calendario, restantes_visibles, reference_date,
        liga_seleccionada, goal_model,
    )

with tab_tabla:
    tab_standings.render(real, proyectada)

with tab_stat:
    tab_stats.render(partidos_temporada, equipos_liga, liga_seleccionada)
