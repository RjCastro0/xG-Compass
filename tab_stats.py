"""
=============================================================================
 TAB: STATS — Stats / Attack / Defense (tables, scatter charts, radars)
=============================================================================
"""
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from team_stats import calcular_stats_equipos, calcular_forma_reciente, calcular_stats_radar


def top5_chart(stats_df, metric, ascending=False, titulo=None):
    """Ranked #/Team/Value list with the top 5 of a metric, sorted."""
    datos = stats_df[["equipo", metric]].sort_values(metric, ascending=ascending).head(5).reset_index(drop=True)
    datos.index = datos.index + 1
    datos = datos.rename(columns={"equipo": "Team", metric: "Value"})
    datos.index.name = "#"
    st.markdown(f"**{titulo or metric}**")
    st.dataframe(
        datos.style.format({"Value": "{:.1f}"}),
        use_container_width=True,
    )


def scatter_cuadrantes(df, x, y, x_label, y_label, titulo, color_scale, x_ref=None, y_ref=None):
    """Scatter with reference lines splitting it into 4 quadrants. Legend with
    the team names (fixed color per team); only the points show on the chart."""
    x_ref = df[x].mean() if x_ref is None else x_ref
    y_ref = df[y].mean() if y_ref is None else y_ref
    base = alt.Chart(df).encode(
        x=alt.X(x, title=x_label), y=alt.Y(y, title=y_label),
        tooltip=["equipo", alt.Tooltip(x, format=".1f"), alt.Tooltip(y, format=".1f")],
    )
    puntos = base.mark_circle(size=130).encode(color=color_scale)
    linea_x = alt.Chart(pd.DataFrame({"x": [x_ref]})).mark_rule(strokeDash=[4, 4], color="gray").encode(x="x")
    linea_y = alt.Chart(pd.DataFrame({"y": [y_ref]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y")
    chart = (linea_x + linea_y + puntos).properties(title=titulo, height=460)
    st.altair_chart(chart, use_container_width=True)


def scatter_regresion(df, x, y, x_label, y_label, titulo, color_scale):
    """Scatter with a linear regression line (altair, no scipy/sklearn needed).
    Legend with the team names (fixed color per team); only the points show on the chart."""
    base = alt.Chart(df).encode(x=alt.X(x, title=x_label), y=alt.Y(y, title=y_label))
    puntos = base.mark_circle(size=130).encode(
        color=color_scale,
        tooltip=["equipo", alt.Tooltip(x, format=".1f"), alt.Tooltip(y, format=".1f")],
    )
    linea = base.transform_regression(x, y).mark_line(color="#FF6B6B")
    chart = (puntos + linea).properties(title=titulo, height=460)
    st.altair_chart(chart, use_container_width=True)


def radar_comparativo(stats_radar, teams_sel, metric_cols, metric_labels, color_scale):
    """Hand-built radar in altair (filled polygon over polar coordinates, scouting-
    report style). Each metric is normalized 0-100% against ALL teams in the league
    (min-max, not a real percentile), so the shape is comparable across different
    team combinations."""
    n = len(metric_cols)
    angulos = [2 * np.pi * i / n - np.pi / 2 for i in range(n)]

    norm = stats_radar.copy()
    for c in metric_cols:
        cmin, cmax = norm[c].min(), norm[c].max()
        norm[c] = (norm[c] - cmin) / (cmax - cmin) if cmax > cmin else 0.5

    grilla_niveles = [0.25, 0.5, 0.75, 1.0]
    grilla = pd.concat([
        pd.DataFrame({
            "x": [r * np.cos(a) for a in angulos] + [r * np.cos(angulos[0])],
            "y": [r * np.sin(a) for a in angulos] + [r * np.sin(angulos[0])],
            "orden": range(n + 1), "nivel": r,
        }) for r in grilla_niveles
    ], ignore_index=True)
    spokes = pd.DataFrame({
        "x": [0] * n + [np.cos(a) for a in angulos],
        "y": [0] * n + [np.sin(a) for a in angulos],
        "eje": list(range(n)) * 2,
        "orden": [0] * n + [1] * n,
    })
    etiquetas = pd.DataFrame({
        "x": [1.32 * np.cos(a) for a in angulos], "y": [1.32 * np.sin(a) for a in angulos],
        "metrica": metric_labels,
    })

    filas, vertices = [], []
    for team in teams_sel:
        row_norm = norm[norm.equipo == team]
        row_raw = stats_radar[stats_radar.equipo == team]
        if row_norm.empty:
            continue
        row_norm, row_raw = row_norm.iloc[0], row_raw.iloc[0]
        for i, c in enumerate(metric_cols):
            a = angulos[i]
            x, y = row_norm[c] * np.cos(a), row_norm[c] * np.sin(a)
            filas.append({"equipo": team, "orden": i, "metrica": metric_labels[i], "x": x, "y": y,
                          "valor": row_raw[c]})
            if len(teams_sel) == 1:
                vertices.append({
                    "x": 1.14 * np.cos(a), "y": 1.14 * np.sin(a),
                    "etiqueta": f"{row_norm[c] * 100:.0f}%",
                })
    df_pol = pd.DataFrame(filas)

    fondo_grilla = alt.Chart(grilla).mark_line(color="gray", strokeWidth=1, strokeDash=[3, 3]).encode(
        x=alt.X("x", axis=None, scale=alt.Scale(domain=[-1.98, 1.98])),
        y=alt.Y("y", axis=None, scale=alt.Scale(domain=[-1.4, 1.4])),
        order="orden", detail="nivel",
    )
    ejes = alt.Chart(spokes).mark_line(color="gray", strokeWidth=1, strokeDash=[3, 3]).encode(
        x="x", y="y", detail="eje", order="orden"
    )
    texto_ejes = alt.Chart(etiquetas).mark_text(fontSize=13, fontWeight="bold", color="white").encode(
        x="x", y="y", text="metrica"
    )
    if df_pol.empty:
        chart = (fondo_grilla + ejes + texto_ejes).properties(width=680, height=480)
    else:
        poligonos = alt.Chart(df_pol).mark_line(
            interpolate="linear-closed", strokeWidth=2.5, fillOpacity=0.35,
        ).encode(
            x="x", y="y", color=color_scale, fill=color_scale, detail="equipo", order="orden",
            tooltip=["equipo", "metrica", alt.Tooltip("valor", format=".1f")],
        )
        puntos = alt.Chart(df_pol).mark_circle(size=70).encode(
            x="x", y="y", color=color_scale, detail="equipo",
            tooltip=["equipo", "metrica", alt.Tooltip("valor", format=".1f")],
        )
        capas = [fondo_grilla, ejes, poligonos, puntos]
        if vertices:
            df_valores = pd.DataFrame(vertices)
            capas.append(
                alt.Chart(df_valores).mark_text(fontSize=14, fontWeight="bold", color="white").encode(
                    x="x", y="y", text="etiqueta",
                )
            )
        capas.append(texto_ejes)
        chart = alt.layer(*capas).properties(width=680, height=480)
    st.altair_chart(chart, use_container_width=False)


def render(partidos_temporada, equipos_liga, liga_seleccionada):
    st.caption(f"Showing: **{liga_seleccionada}**")
    stats_equipos = calcular_stats_equipos(partidos_temporada, equipos_liga, liga=liga_seleccionada)
    stats_equipos["Faltas/partido"] = stats_equipos["Faltas"] / stats_equipos["PJ"]

    seccion = st.selectbox("Section", ["Stats", "Attack", "Defense"], key="sel_seccion_stats")

    if seccion == "Stats":
        st.subheader("Team statistics (current season)")
        columnas = ["equipo", "PJ", "xG/partido", "xGA/partido", "Goles", "GA", "Win%",
                    "Clean sheets", "%Tiros anotados", "%Tiros a puerta", "Faltas/partido",
                    "Amarillas/partido", "Rojas"]
        tabla = stats_equipos[columnas].sort_values("xG/partido", ascending=False)
        tabla = tabla.rename(columns={
            "equipo": "Team", "PJ": "MP", "xG/partido": "xG/match", "xGA/partido": "xGA/match",
            "Goles": "Goals", "%Tiros anotados": "%Shots scored", "%Tiros a puerta": "%Shots on target",
            "Faltas/partido": "Fouls/match", "Amarillas/partido": "Yellows/match", "Rojas": "Reds",
        })
        st.dataframe(
            tabla.style.format({
                "xG/match": "{:.2f}", "xGA/match": "{:.2f}", "Win%": "{:.1f}",
                "%Shots scored": "{:.1f}", "%Shots on target": "{:.1f}",
                "Fouls/match": "{:.1f}", "Yellows/match": "{:.2f}",
                "Goals": "{:.0f}", "GA": "{:.0f}", "Reds": "{:.0f}",
            }),
            use_container_width=True, height=740, hide_index=True,
        )

    elif seccion == "Attack":
        st.subheader("Leaders — Attack")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            top5_chart(stats_equipos, "xG/partido", titulo="Most xG/match")
        with col2:
            top5_chart(stats_equipos, "Tiros/partido", titulo="Most shots/match")
        with col3:
            top5_chart(stats_equipos, "Goles", titulo="Most goals")
        with col4:
            top5_chart(stats_equipos, "xGOT/partido", titulo="Most xGOT/match")

    elif seccion == "Defense":
        st.subheader("Leaders — Defense")
        st.caption("xGA, Goals and Shots conceded: lower = better defense.")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            top5_chart(stats_equipos, "xGA/partido", ascending=True, titulo="Best xGA/match")
        with col2:
            top5_chart(stats_equipos, "GA", ascending=True, titulo="Fewest goals conceded")
        with col3:
            top5_chart(stats_equipos, "Tiros permitidos/partido", ascending=True,
                       titulo="Fewest shots conceded")
        with col4:
            top5_chart(stats_equipos, "Clean sheets", titulo="Most clean sheets")

    # ---------------------------------------------------------------- charts
    # fixed domain -> every team keeps the SAME color across every chart,
    # whether all 20 show up or not (category20 assigns by position in the domain)
    color_equipos = alt.Color(
        "equipo:N", title="Team",
        scale=alt.Scale(domain=equipos_liga, scheme="category20"),
        legend=alt.Legend(orient="right", columns=1, labelFontSize=11, symbolSize=90),
    )

    if seccion == "Attack":
        st.divider()
        forma5 = calcular_forma_reciente(partidos_temporada, equipos_liga, n=5, liga=liga_seleccionada)

        c1, c2 = st.columns(2)
        with c1:
            scatter_cuadrantes(
                forma5, "puntos_last5", "xgd_last5",
                "Form (points last 5)", "Quality (xGD last 5)",
                "Form vs Quality — last 5 matches", color_equipos, y_ref=0,
            )
            st.caption("Top-right: good results AND good play. Bottom-right: good results "
                       "but lucky (negative xGD) — watch for possible regression.")
        with c2:
            # 90 min ≈ 1 league match -> reuses the already-computed /match metrics
            scatter_cuadrantes(
                stats_equipos, "xG/partido", "Tiros/partido",
                "xG / 90 min", "Shots / 90 min",
                "Shots/90 vs xG/90", color_equipos,
            )
            st.caption("Top-right: high shot volume and good quality. Bottom-right: good "
                       "chance quality but low shot volume.")

        stats_equipos["xG_total"] = stats_equipos["xG/partido"] * stats_equipos["PJ"]
        scatter_regresion(stats_equipos, "xG_total", "Goles", "xG (season total)", "Goals",
                           "xG vs Goals — Performance", color_equipos)

        st.markdown("**Attack radar — compare up to 3 teams**")
        equipos_radar_at = st.multiselect(
            "Teams", equipos_liga, default=equipos_liga[:2],
            max_selections=3, key="sel_radar_ataque",
        )
        cols_at = ["xg", "goles", "xgot", "pases_ultimo_tercio", "posesion", "regates"]
        stats_radar_at = calcular_stats_radar(partidos_temporada, equipos_liga, cols_at, liga=liga_seleccionada)
        radar_comparativo(
            stats_radar_at, equipos_radar_at, cols_at,
            ["xG", "Goals", "xGOT", "Final third passes", "Possession", "Dribbles"],
            color_equipos,
        )

    elif seccion == "Defense":
        st.divider()
        stats_equipos["xGA_total"] = stats_equipos["xGA/partido"] * stats_equipos["PJ"]
        stats_equipos["tiros_permitidos_total"] = stats_equipos["Tiros permitidos/partido"] * stats_equipos["PJ"]
        stats_equipos["tiros_por_gol_c"] = stats_equipos["tiros_permitidos_total"] / stats_equipos["GA"]
        stats_equipos["xg_por_tiro_c"] = stats_equipos["xGA_total"] / stats_equipos["tiros_permitidos_total"]

        c1, c2 = st.columns(2)
        with c1:
            scatter_regresion(stats_equipos, "xGA_total", "GA", "xGA (season total)",
                               "Goals conceded", "Goals conceded vs xGA", color_equipos)
        with c2:
            st.markdown("**Defensive radar — compare up to 3 teams**")
            equipos_radar_def = st.multiselect(
                "Teams", equipos_liga, default=equipos_liga[:2],
                max_selections=3, key="sel_radar_defensivo",
            )
            cols_def = ["intercepciones", "duelos_ganados", "paradas", "despejes",
                        "entradas_ganadas", "posesiones_ganadas_ultimo_tercio"]
            stats_radar_def = calcular_stats_radar(partidos_temporada, equipos_liga, cols_def, liga=liga_seleccionada)
            radar_comparativo(
                stats_radar_def, equipos_radar_def, cols_def,
                ["Interceptions", "Duels won", "Saves", "Clearances",
                 "Successful tackles", "Recoveries in Attacking 3rd"],
                color_equipos,
            )

        c3, c4 = st.columns(2)
        with c3:
            scatter_cuadrantes(
                stats_equipos, "Tiros permitidos/partido", "xGA/partido",
                "Shots conceded / match", "xG conceded / match",
                "Volume vs quality of shots conceded", color_equipos,
            )
        with c4:
            forma5_def = calcular_forma_reciente(partidos_temporada, equipos_liga, n=5, liga=liga_seleccionada)
            comp = stats_equipos.merge(forma5_def[["equipo", "xga_prom_last5"]], on="equipo")
            scatter_cuadrantes(
                comp, "xGA/partido", "xga_prom_last5",
                "xGA/match (season)", "xGA/match (last 5)",
                "Recent xGA vs full season", color_equipos,
            )
            st.caption("Above the implicit diagonal: defending worse lately than the season "
                       "average. Below: improving.")
