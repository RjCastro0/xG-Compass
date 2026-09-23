"""
=============================================================================
 TAB: FIXTURE DIFFICULTY — team x matchday grid, green (easy) -> red (hard)
=============================================================================
"""
import numpy as np
import pandas as pd
import streamlit as st

from difficulty import score_dificultad_bruto

SNAPSHOT_DIFICULTAD_PATH = "dificultad_snapshot.csv"


def indice_1_a_5(score, vmin, vmax):
    """Maps the raw (unbounded) score to a bounded index 1.0 (easy) - 5.0 (hard)."""
    if pd.isna(score):
        return None
    t = (score - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    t = min(max(t, 0.0), 1.0)
    return 1 + 4 * t


def cargar_snapshot_dificultad():
    """Indices from the last load (team, matchday) -> value. Empty on first run."""
    try:
        snap = pd.read_csv(SNAPSHOT_DIFICULTAD_PATH)
        return {(r.equipo, r.jornada): r.indice for r in snap.itertuples()}
    except FileNotFoundError:
        return {}


def guardar_snapshot_dificultad(valores):
    """valores: dict (team, matchday) -> index. Overwrites the snapshot for the next comparison."""
    filas = [{"equipo": eq, "jornada": jr, "indice": v} for (eq, jr), v in valores.items()]
    pd.DataFrame(filas).to_csv(SNAPSHOT_DIFICULTAD_PATH, index=False)


def construir_grid_dificultad(prior_model, partidos_temporada, calendario_restante, jornadas,
                               reference_date, liga_seleccionada, equipos_liga):
    """
    Two aligned DataFrames (rows = team, columns = matchday):
      - texto:  "OPP (H)" + 1-5 index + delta vs the last load, for the cell display
      - score:  the raw (unbounded) score, only used to compute percentiles and color

    Receives the calendar ALREADY FILTERED to pending matches (fixtures_restantes),
    not the full calendar -> so a matchday with one postponed match doesn't drag
    the rest of that team's already-played matches back into the grid.
    """
    texto = pd.DataFrame(index=equipos_liga, columns=jornadas, dtype=object)
    score = pd.DataFrame(index=equipos_liga, columns=jornadas, dtype=float)
    rival_info = {}

    for jornada in jornadas:
        for _, row in calendario_restante[calendario_restante.jornada == jornada].iterrows():
            for equipo, rival, es_local in [
                (row.equipo_local, row.equipo_visitante, True),
                (row.equipo_visitante, row.equipo_local, False),
            ]:
                if equipo not in score.index:
                    continue
                s = score_dificultad_bruto(equipo, rival, es_local, prior_model,
                                            partidos_temporada, reference_date, liga=liga_seleccionada)
                score.loc[equipo, jornada] = s
                rival_info[(equipo, jornada)] = f"{rival}({'H' if es_local else 'A'})"

    # percentiles over EVERYTHING being shown -> a 1-5 index comparable across
    # matchdays, and one extreme fixture doesn't skew the rest of the scale
    valores_validos = score.values[~np.isnan(score.values)]
    vmin, vmax = np.percentile(valores_validos, 5), np.percentile(valores_validos, 95)

    snapshot_anterior = cargar_snapshot_dificultad()
    snapshot_nuevo = {}

    for equipo in score.index:
        for jornada in score.columns:
            s = score.loc[equipo, jornada]
            if pd.isna(s):
                continue
            idx = indice_1_a_5(s, vmin, vmax)
            snapshot_nuevo[(equipo, jornada)] = idx

            anterior = snapshot_anterior.get((equipo, jornada))
            delta = (idx - anterior) if anterior is not None else None
            if delta is not None and round(delta, 1) != 0:
                flecha = "▼" if delta < 0 else "▲"
                texto.loc[equipo, jornada] = f"{rival_info[(equipo, jornada)]}  {idx:.1f}  {flecha}{abs(delta):.1f}"
            else:
                texto.loc[equipo, jornada] = f"{rival_info[(equipo, jornada)]}  {idx:.1f}"

    guardar_snapshot_dificultad(snapshot_nuevo)
    return texto, score, vmin, vmax


def color_por_dificultad(score, vmin, vmax):
    """
    Interpolates green -> yellow -> red by difficulty.
    Green = easy fixture (weak opponent)
    Red   = hard fixture (strong opponent)
    """
    if pd.isna(score):
        return "background-color: #1a1a1a; color: #666"

    # normalize 0 (easiest) to 1 (hardest)
    t = (score - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    t = min(max(t, 0.0), 1.0)

    if t < 0.5:  # green -> yellow
        ratio = t / 0.5
        r = int(30 + (200 - 30) * ratio)
        g = int(150 + (180 - 150) * ratio)
        b = int(60 + (60 - 60) * ratio)
    else:        # yellow -> red
        ratio = (t - 0.5) / 0.5
        r = int(200 + (190 - 200) * ratio)
        g = int(180 + (40 - 180) * ratio)
        b = 60 - int(20 * ratio)

    # white text on dark backgrounds, black on the lighter ones in the middle
    luminancia = 0.299 * r + 0.587 * g + 0.114 * b
    color_texto = "#000000" if luminancia > 140 else "#ffffff"
    return f"background-color: rgb({r},{g},{b}); color: {color_texto}"


def render(prior_model, partidos_temporada, restantes_visibles, reference_date,
           liga_seleccionada, equipos_liga):
    st.subheader("Difficulty of upcoming fixtures")
    st.caption("Index from 1.0 (easiest) to 5.0 (hardest). (H) home, (A) away.")

    jornadas_disponibles = sorted(restantes_visibles.jornada.unique())
    n_jornadas = st.slider("Matchdays to show", 3, 12, 8)
    jornadas = jornadas_disponibles[:n_jornadas]

    texto, score, vmin, vmax = construir_grid_dificultad(
        prior_model, partidos_temporada, restantes_visibles, jornadas,
        reference_date, liga_seleccionada, equipos_liga,
    )

    estilos = pd.DataFrame(
        [[color_por_dificultad(score.iloc[i, j], vmin, vmax) for j in range(score.shape[1])]
         for i in range(score.shape[0])],
        index=score.index, columns=score.columns,
    )

    # sort teams by average difficulty (easiest schedule on top)
    orden = score.mean(axis=1).sort_values().index
    texto_ord = texto.loc[orden]
    estilos_ord = estilos.loc[orden]
    texto_ord.columns = [f"MD{j}" for j in texto_ord.columns]
    estilos_ord.columns = texto_ord.columns

    st.dataframe(
        texto_ord.style.apply(lambda _: estilos_ord, axis=None),
        use_container_width=True,
        height=760,
    )

    with st.expander("How it's calculated"):
        st.markdown("""
        - **Opponent's historical rating**: their attacking output in the
          venue they'll play (`away_attack` if they're visiting, `home_attack`
          if they're at home).
        - **Current-form adjustment**: xG generated this season and the xG
          the opponent has been conceding, both as a ratio against the
          league average.

        `difficulty = opponent_historical_attack − (current_form) × (opponent_xG_conceded/league_xG)`

        The number in the cell is the **1.0 (easiest) to 5.0 (hardest)
        index**, computed by percentiles over all the matchdays shown —
        one extreme fixture (e.g. a team on a hot streak) doesn't skew the
        rest of the scale.
        """)
