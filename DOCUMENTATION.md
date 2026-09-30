# xG-Compass

Dashboard de predicción de fútbol para 5 ligas europeas (LaLiga, Premier League, Serie A, Bundesliga, Ligue 1): predice partidos, calcula la dificultad de los próximos fixtures y proyecta la tabla de posiciones final mediante simulación Monte Carlo, a partir de un modelo de goles Dixon-Coles.

---

## 1. Qué hace

| Capacidad | Cómo |
|---|---|
| Predecir un partido | Distribución de Poisson por equipo (ataque/defensa) + corrección Dixon-Coles para marcadores bajos |
| Medir la dificultad de un fixture | Rating histórico del rival, ajustado por la forma ofensiva propia y el xG que el rival viene concediendo |
| Proyectar la tabla final | 20.000 simulaciones Monte Carlo del calendario real restante, arrancando desde los puntos ya obtenidos |
| Comparar equipos | Radares normalizados (0-100% contra la liga) de métricas de ataque y defensa |

---

## 2. Metodología

### 2.1 Dos motores de goles, dos roles distintos

**`TimeDecayGoalModel`** — prior histórico (solo goles, 2000-2026, multi-liga)
- Rating de 4 valores por equipo: `home_attack`, `home_defense`, `away_attack`, `away_defense` (1.0 = promedio de liga).
- Decaimiento exponencial: cada partido pierde la mitad de su peso cada **730 días**.
- Shrinkage hacia el promedio de liga para equipos con poco historial (ej. recién ascendidos).
- Uso: único insumo del score de dificultad de fixtures.

**`HybridGoalModel`** — motor de predicción (el que se usa para partidos y Monte Carlo)
- Blend del prior histórico con un **compuesto de la temporada actual**: xG, xA, tiros, tiros a puerta, ocasiones claras, % de tiros a puerta (ataque); xGA, goles concedidos, tiros a puerta concedidos, PPDA, conversión permitida (defensa).
- Cada métrica pondera según su **correlación medida** con los goles del partido siguiente (ej. xG r=0.176, xGA r=0.35 dentro de sus respectivos compuestos).
- Decaimiento propio de la forma de temporada: la mitad del peso se pierde cada **5 partidos jugados** (independiente del decaimiento de 730 días del prior).
- Blend histórico ↔ temporada actual ponderado por `n/(n+k)`, con `k` distinto para ataque (k=3, converge rápido) y defensa (k=30, señal más ruidosa, converge más lento).
- **Corrección Dixon-Coles (τ):** ajusta la matriz de probabilidad de marcador en las celdas 0-0, 1-0, 0-1 y 1-1, donde un Poisson independiente simple sobre/subestima sistemáticamente. El parámetro ρ se ajusta por grid search maximizando la log-verosimilitud sobre los partidos reales de la temporada en curso.

### 2.2 Validación (walk-forward, La Liga 26-27, 40 partidos)

Cada predicción usa **solo** información anterior a la fecha del partido — sin ver el futuro.

| Versión del modelo | Acierto 1X2 |
|---|---|
| Solo prior histórico (goles) | 53.8% |
| + compuesto de temporada actual | 59.0% |
| + corrección Dixon-Coles (ρ ajustado) | **64.1%** ← en producción |

### 2.3 Score de dificultad de fixture

```
dificultad = ataque_histórico_del_rival − (mi_forma_actual) × (xG_concedido_rival / xG_liga)
```

- `ataque_histórico_del_rival`: rating de `TimeDecayGoalModel` en la faceta que juega (`away_attack` si visita, `home_attack` si es local).
- `mi_forma_actual`: compuesto ponderado de mi xG generado y mis goles reales esta temporada (r=0.530 y r=0.455 respectivamente contra goles futuros).
- Validado contra los goles reales del siguiente partido: usar solo el rating histórico da r=+0.17 (dirección correcta, señal débil); con el ajuste de forma actual sube a r=-0.38.
- Se normaliza a un **índice de 1.0 (más fácil) a 5.0 (más difícil)** por percentiles sobre las jornadas mostradas, para que un solo fixture extremo no rompa la escala del resto.

### 2.4 Simulación Monte Carlo de temporada

- Arranca desde los **puntos reales ya obtenidos**, no desde cero.
- Juega el **calendario real restante** (no uno sintético), partido a partido, muestreando de la distribución de marcador completa (con Dixon-Coles).
- 20.000 repeticiones → probabilidad de título, de Top 4, de descenso, y posición mediana por equipo.

---

## 3. Arquitectura

Archivos planos, sin subcarpetas.

### Backend

| Módulo | Contenido |
|---|---|
| `config.py` | Rutas de datos, hiperparámetros, diccionario `ALIAS` de normalización de nombres de equipo/liga |
| `data_loader.py` | Carga y unificación del histórico + temporada actual + calendario multi-liga |
| `goal_models.py` | `TimeDecayGoalModel`, `HybridGoalModel`, Poisson, corrección Dixon-Coles |
| `difficulty.py` | Score de dificultad de fixture (sección 2.3) |
| `schedule.py` | Tabla real actual, fixtures restantes, detección de aplazados, a qué jornada pertenece cada partido |
| `predictions.py` | Ratings, predicción de partido puntual, análisis/predicciones por jornada |
| `simulation.py` | Monte Carlo de temporada y tabla proyectada |
| `team_stats.py` | Estadísticas de equipo para el tab Stats (tablas y radares) |
| `data_ingestion.py` | Ingesta de CSV crudos (FBref) hacia histórico/calendario — soporte de `agregar_datos.ipynb` |

### Frontend (Streamlit)

| Módulo | Tab |
|---|---|
| `App.py` | Punto de entrada: carga/fit cacheados por liga, sidebar, selector de liga |
| `tab_difficulty.py` | Fixture Difficulty — grid equipo × jornada |
| `tab_matchdays.py` | Matchdays — jornada anterior jugada / próxima por jugar |
| `tab_standings.py` | Standings — tabla real vs. proyectada |
| `tab_stats.py` | Stats — tabla general, top 5 ataque/defensa, scatter charts, radares |

### Datos (no versionados, van en la raíz del repo)

| Archivo | Contenido |
|---|---|
| `partidos.csv` | Partidos de la temporada 2026-27, todas las ligas, stats completas por partido |
| `calendario_partidos.csv` | Fixture completo por liga (jornada + fecha programada) |
| `all_time_df_actualizado.csv` | Histórico 2000-2026, multi-liga (~60.000 partidos) |
| `dificultad_snapshot.csv` | Se genera solo — guarda el índice de dificultad de la carga anterior para mostrar el delta |

---

## 4. Normalización de equipos (`ALIAS`)

Cada fuente de datos usa una convención de nombres distinta para el mismo equipo (ej. `"Inter"` en el histórico vs. `"Inter Milan"` en `partidos.csv`; `"Freiburg"` vs. `"Friburgo"`). El diccionario `ALIAS` en `config.py` normaliza todo hacia el nombre que usa `partidos.csv`.

**Regla crítica:** nunca crear una entrada `A → B` si ya existe `B → A` (o cualquier ciclo) — se cancelan entre sí y dejan el nombre original sin normalizar. `verificar_alias_conflictos()` detecta esto automáticamente en cada ingesta.

Cuando un equipo nuevo no matchea entre archivos, `data_ingestion.py` lo señala solo; se agrega el alias con `agregar_alias_permanente()` (lo escribe directo en `config.py`, sin editar el archivo a mano).

---

## 5. Agregar datos nuevos

`agregar_datos.ipynb` automatiza:

1. **Temporada histórica completa** (ya jugada) → `all_time_df_actualizado.csv`, vía `agregar_temporada_historico()`.
2. **Calendario de una liga** (jornada + fechas, temporada en curso) → `calendario_partidos.csv`, vía `agregar_liga_calendario()`.

Ambas funciones parten de un CSV crudo formato FBref (`Wk,Day,Date,Time,Home,Score,Away,...`), detectan duplicados automáticamente, y avisan si algún nombre de equipo no matchea con `partidos.csv` (candidato a alias faltante).

Los **resultados de la jornada actual** (`partidos.csv`) se cargan aparte, con el formato de estadísticas completas por partido (365scores), mergeando por fecha+equipos y descartando duplicados.

---

## 6. Instalación y deploy

```bash
pip install -r requirements.txt
streamlit run App.py
```

Necesita los 3 CSV de datos (sección 3) en la misma carpeta que `App.py`.

**Streamlit Cloud:** además de `streamlit`, `pandas`, `numpy` y `altair`, `requirements.txt` debe incluir `matplotlib` (lo usa el degradé de color de la tabla proyectada en Standings) — el entorno de nube no trae dependencias opcionales de pandas preinstaladas como sí puede pasar en una máquina local.

---

## 7. Actualización de datos durante la temporada

Al final de cada jornada, por liga:

1. Subir el CSV crudo de resultados (365scores) → se mergea en `partidos.csv`.
2. El modelo, la dificultad de fixtures y la proyección se recalculan solos en el próximo `Reload data` del sidebar (limpia la caché de Streamlit) — no hace falta reiniciar nada más.

Premier League, Serie A, Bundesliga y Ligue 1 necesitan además que `calendario_partidos.csv` cubra sus jornadas ya jugadas desde la jornada 1 para que "Jornada anterior" (tab Matchdays) pueda identificar el número exacto de jornada; si falta ese tramo, el tab muestra automáticamente los últimos resultados por fecha en su lugar, sin romperse.
