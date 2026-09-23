"""
=============================================================================
 TAB: STANDINGS — current real table vs. projected table (Monte Carlo)
=============================================================================
"""
import streamlit as st


def render(real, proyectada):
    col_izq, col_der = st.columns(2)

    with col_izq:
        st.subheader("Current table (actual)")
        tabla_real = real.copy()
        tabla_real.index = tabla_real.index + 1
        tabla_real = tabla_real[["team", "PJ", "Pts", "GF", "GA", "GD"]].rename(columns={
            "team": "Team", "PJ": "MP",
        })
        st.dataframe(tabla_real, use_container_width=True, height=740)

    with col_der:
        st.subheader("Projected table (end of season)")
        tabla_proy = proyectada.copy()
        tabla_proy.index = tabla_proy.index + 1
        tabla_proy = tabla_proy.rename(columns={
            "team": "Team", "exp_Pts": "Exp. Pts", "exp_GD": "Exp. GD",
            "title_pct": "Title %", "top4_pct": "Top 4 %",
            "relegation_pct": "Relegation %", "median_finish": "Median finish",
        })
        st.dataframe(
            tabla_proy.style.background_gradient(subset=["Title %"], cmap="Greens")
                            .background_gradient(subset=["Relegation %"], cmap="Reds")
                            .format({"Title %": "{:.1f}", "Top 4 %": "{:.1f}",
                                     "Relegation %": "{:.1f}"}),
            use_container_width=True, height=740,
        )

    st.caption(
        "The projected table starts from the real points already earned and "
        "simulates the actual remaining schedule, averaging thousands of seasons."
    )
