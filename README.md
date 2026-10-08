# xG-Compass
Football Match &amp; Standings Predictions

# Football Match & Standings Prediction Dashboard

Dashboard en Streamlit que predice partidos, calcula la dificultad de los próximos fixtures y proyecta la tabla de posiciones (Monte Carlo) para 5 ligas europeas, a partir de un modelo híbrido de goles y xG (Dixon-Coles).
Enlace de la aplicacion: https://xgcompass.streamlit.app/ 

**Ligas soportadas:** La Liga, Premier League, Serie A, Bundesliga, Ligue 1.

## Tabs

| Tab | Qué muestra |
|---|---|
| **Fixture Difficulty** | Grid equipo × jornada, verde (fácil) → rojo (difícil), con índice 1.0–5.0 |
| **Matchdays** | Resultados de la última jornada jugada (xG, xGD, rendimiento) y predicciones de la próxima |
| **Standings** | Tabla real actual vs. tabla proyectada a fin de temporada (20.000 simulaciones Monte Carlo) |
| **Stats** | Estadísticas por equipo, top 5 de ataque/defensa, y radares comparativos (hasta 3 equipos) |

## Modelo

Dos motores con roles distintos:

- **`TimeDecayGoalModel`** (prior histórico): ratings de ataque/defensa con decaimiento exponencial (half-life 730 días) sobre 2000–2026, multi-liga. Insumo del score de dificultad de fixtures.
- **`HybridGoalModel`** (motor de predicción): blend del prior histórico con un compuesto multi-métrica de la temporada actual (xG, xA, tiros, ocasiones claras, xGA, PPDA...), más la corrección de Dixon-Coles para marcadores bajos.

Validado walk-forward (La Liga 26-27, 40 partidos):

| Versión | Acierto 1X2 |
|---|---|
| Solo goles históricos | 53.8% |
| + compuesto de temporada actual | 59.0% |
| + corrección Dixon-Coles (ρ ajustado) | **64.1%** ← el que usa la app |

## Estructura del proyecto

Todo en archivos planos, sin subcarpetas.

**Backend (lógica y datos):**

```
config.py           rutas, hiperparámetros, diccionario de alias de equipos/ligas
data_loader.py       carga y unificación del histórico + temporada actual + calendario
goal_models.py       TimeDecayGoalModel, HybridGoalModel, Poisson/Dixon-Coles
difficulty.py        score de dificultad de fixture (1-5)
schedule.py          standings reales, fixtures restantes, aplazados, detección de jornadas
predictions.py       ratings, predicción de partido puntual, predicciones/análisis por jornada
simulation.py        Monte Carlo de temporada y tabla proyectada
team_stats.py        estadísticas de equipo para el tab Stats (tablas y radares)
data_ingestion.py    ingesta de CSV crudos (FBref) hacia histórico/calendario
```

**Frontend (Streamlit):**

```
App.py               punto de entrada: carga/fit cacheados, sidebar, selector de liga
tab_difficulty.py     tab Fixture Difficulty
tab_matchdays.py      tab Matchdays
tab_standings.py      tab Standings
tab_stats.py          tab Stats (incluye los gráficos y radares)
```

**Datos (no versionados en el repo, van en la misma carpeta):**

```
partidos.csv                    partidos de la temporada 2026-27, todas las ligas
calendario_partidos.csv         fixture completo por liga (jornada + fecha)
all_time_df_actualizado.csv     histórico 2000-2026, multi-liga
dificultad_snapshot.csv         se genera solo (guarda el índice de dificultad anterior)
```

## Instalación y uso

```bash
pip install -r requirements.txt
streamlit run App.py
```

Necesita los 3 CSV de datos (`partidos.csv`, `calendario_partidos.csv`, `all_time_df_actualizado.csv`) en la misma carpeta.

## Agregar datos nuevos

`agregar_datos.ipynb` automatiza el proceso de cargar una temporada histórica completa o el calendario de una liga, a partir de un CSV crudo (formato FBref). Detecta duplicados, valida que los nombres de equipo coincidan entre archivos, y avisa si falta agregar un alias — sin tener que tocar código a mano. Ver las instrucciones dentro del propio notebook.

## Notas

- El modelo se ajusta una sola vez por liga y queda cacheado (`st.cache_resource`); el botón **Reload data** del sidebar limpia la caché si se actualizaron los CSV.
- La normalización de nombres de equipo (`ALIAS` en `config.py`) es crítica: distintas fuentes usan convenciones distintas (ej. "Inter" vs "Inter Milan", "Freiburg" vs "Friburgo"). Cualquier equipo nuevo que no matchee entre `partidos.csv` y el histórico/calendario hay que agregarlo ahí.
- Deploy en Streamlit Cloud: además de lo que ya está en `requirements.txt`, asegurate de que `matplotlib` esté incluido (lo usa `Standings` para el degradé de color de la tabla proyectada).
