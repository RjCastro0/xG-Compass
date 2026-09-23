"""
=============================================================================
 TAB: MATCHDAYS — previous matchday played | upcoming matchday
=============================================================================
"""
import streamlit as st

from schedule import detectar_jornadas, jornada_de_cada_partido
from predictions import analisis_jornada_jugada, analisis_ultimos_partidos, predicciones_jornada


def render(partidos_temporada, calendario, restantes_visibles, reference_date,
           liga_seleccionada, goal_model):
    jornada_ant, jornada_prox = detectar_jornadas(partidos_temporada, calendario, reference_date,
                                                   liga=liga_seleccionada)
    jornadas_jugadas = sorted(
        jornada_de_cada_partido(partidos_temporada, calendario, liga=liga_seleccionada)
        .jornada.unique().astype(int)
    )
    jornadas_futuras = sorted(restantes_visibles.jornada.unique().astype(int))

    col_izq, col_der = st.columns(2)

    # ---------------------------------------------------------------- past
    with col_izq:
        st.subheader("Previous matchday")
        if not jornadas_jugadas:
            st.caption(
                "This league's calendar doesn't cover the first matchdays already "
                "played yet, so matches can't be grouped by matchday number — the "
                "latest matches played, by date, are shown instead."
            )
            analisis = analisis_ultimos_partidos(partidos_temporada, liga=liga_seleccionada, n=10)
            if analisis.empty:
                st.info("No matches played yet for this league in partidos.csv.")
            else:
                for r in analisis.itertuples():
                    st.markdown(f"**{r.local} {r.goles_local} - {r.goles_visitante} {r.visitante}**")
                    st.caption(
                        f"{r.fecha} &nbsp;·&nbsp; xG: {r.xg_local:.2f} - {r.xg_visitante:.2f} &nbsp;·&nbsp; "
                        f"xGD: {r.xgd_local:+.2f} / {r.xgd_visitante:+.2f} &nbsp;·&nbsp; "
                        f"Performance: {r.sobre_local:+.2f} / {r.sobre_visitante:+.2f}",
                        unsafe_allow_html=True,
                    )
                    st.divider()
        else:
            j_ant = st.selectbox(
                "Matchday", jornadas_jugadas[::-1],
                index=0 if jornada_ant not in jornadas_jugadas else jornadas_jugadas[::-1].index(jornada_ant),
                key="sel_jornada_ant",
            )
            analisis = analisis_jornada_jugada(partidos_temporada, calendario, j_ant, liga=liga_seleccionada)

            if analisis.empty:
                st.info("No matches played in this matchday.")
            else:
                for r in analisis.itertuples():
                    st.markdown(f"**{r.local} {r.goles_local} - {r.goles_visitante} {r.visitante}**")
                    st.caption(
                        f"xG: {r.xg_local:.2f} - {r.xg_visitante:.2f} &nbsp;·&nbsp; "
                        f"xGD: {r.xgd_local:+.2f} / {r.xgd_visitante:+.2f} &nbsp;·&nbsp; "
                        f"Performance: {r.sobre_local:+.2f} / {r.sobre_visitante:+.2f}",
                        unsafe_allow_html=True,
                    )
                    st.divider()

        with st.expander("What the metrics mean"):
            st.markdown("""
            - **xGD** (`xG − xGA`): match dominance. Positive = created chances
              of higher quality than it conceded, beyond how the scoreline ended up.
            - **Performance** (`goals − xG`): positive = scored more goals than
              its chances justified. Sustained high values tend to revert to the
              mean — it's a sign a scoring streak may not be sustainable.
            """)

    # -------------------------------------------------------------- upcoming
    with col_der:
        st.subheader("Upcoming matchday")
        if not jornadas_futuras:
            st.info("No matches left to play.")
        else:
            default_prox = jornada_prox if jornada_prox in jornadas_futuras else jornadas_futuras[0]
            j_prox = st.selectbox(
                "Matchday", jornadas_futuras,
                index=jornadas_futuras.index(default_prox),
                key="sel_jornada_prox",
            )
            preds = predicciones_jornada(goal_model, calendario, partidos_temporada,
                                          j_prox, reference_date, liga=liga_seleccionada)

            if preds.empty:
                st.info("No matches pending in this matchday.")
            else:
                for r in preds.itertuples():
                    xscore_redondeado = f"{round(r.xg_local)}-{round(r.xg_visitante)}"
                    st.markdown(f"**xScore: {r.local} {xscore_redondeado.replace('-', ' - ')} {r.visitante}**")
                    st.caption(f"{r.p_local:.1%} | Draw {r.p_empate:.1%} | {r.p_visitante:.1%}")
                    st.divider()
