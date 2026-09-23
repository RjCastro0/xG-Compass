"""
=============================================================================
 CONFIG — constantes del proyecto, rutas de archivos y diccionario de alias
=============================================================================
Todo lo que otros módulos necesitan como "settings" vive acá: hiperparámetros
del modelo, rutas de los CSV, la lista fija de equipos de La Liga 26-27, y el
diccionario de normalización de nombres de equipo/liga (ALIAS).

Cada vez que aparece una inconsistencia de nombre nueva (ej. "Getafe CF" vs
"Getafe"), se agrega a ALIAS. Aplicar SIEMPRE normalizar_equipos() antes de
comparar/matchear equipos entre archivos.
"""
import os

# --- hiperparámetros del prior histórico (TimeDecayGoalModel) ---
HALF_LIFE_DAYS = 730          # partidos pierden la mitad de su peso cada ~2 años
SHRINKAGE_STRENGTH = 10.0     # equipos con poco historial se acercan al promedio de liga
N_SIMULATIONS = 20_000        # Monte Carlo de temporada (bajar para desarrollo rápido)

# --- rutas de datos ---
RAW_HIST_PATH = "all_time_df_actualizado.csv"
PARTIDOS_CSV_PATH = "partidos.csv"
CALENDARIO_CSV_PATH = "calendario_partidos.csv"

TEAMS_2026_2027 = [
    "Alavés", "Athletic Club", "Atlético de Madrid", "Barcelona", "Real Betis",
    "Celta de Vigo", "RC Deportivo", "Elche", "Espanyol", "Getafe", "Levante",
    "Málaga", "Osasuna", "Racing de Santander", "Rayo Vallecano", "Real Madrid",
    "Real Sociedad", "Sevilla", "Valencia", "Villarreal",
]
assert len(set(TEAMS_2026_2027)) == 20

# Nombre de liga tal como aparece en all_time_df_actualizado.csv (columna League)
# -> nombre tal como aparece en partidos.csv (columna liga). Se usa SOLO para
# validar equipos contra partidos.csv; el histórico conserva su propio nombre
# de liga (con guion / typo incluido) para no romper temporadas ya cargadas.
LEAGUE_NAME_MAP_HIST_TO_ACTUAL = {
    "La-Liga": "La Liga",
    "Premier-League": "Premier League",
    "Ligue-1": "Ligue 1",
    "Seria-A": "Serie A",
    "Bundesliga": "Bundesliga",
    "Eredivisie": "Eredivisie",
    "Primeira-Liga": "Primeira Liga",
}


# =============================================================================
# DICCIONARIO DE ALIAS (estándar de nombres, ver ways-of-working.md)
# =============================================================================
ALIAS = {
    "Getafe CF": "Getafe",
    "Deportivo": "RC Deportivo", "Dep. A Coruña": "RC Deportivo",
    "Málaga CF": "Málaga",
    "Atlético Madrid": "Atlético de Madrid",
    "R. Racing Club": "Racing de Santander", "Racing Club": "Racing de Santander",
    "Racing Sant": "Racing de Santander",
    "RCD Espanyol": "Espanyol",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "Levante UD": "Levante",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Man City": "Manchester City",
    "Man Utd": "Manchester United", "Manchester Utd": "Manchester United",
    "Gladbach": "Borussia Mönchengladbach", "M'Gladbach": "Borussia Mönchengladbach",
    "Dortmund": "Borussia Dortmund",
    "Frankfurt": "Eintracht Frankfurt", "Eint Frankfurt": "Eintracht Frankfurt",
    "Augsburg": "FC Augsburg",
    "Freiburg": "Friburgo",
    "Hamburger SV": "Hamburgo SV",
    "Leverkusen": "Bayer Leverkusen",
    "Köln": "1. FC Köln",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Nottingham": "Forest",
    "Elversberg": "SV 07 Elversberg",
    "St. Pauli": "St Pauli",
    "Paderborn 07": "SC Paderborn 07",
    "Marseille": "Olympique Marsella",
    "Nice": "Niza",
    "Strasbourg": "Estrasburgo",
    "Paris SG": "PSG", "Paris S-G": "PSG",
    "Milan": "AC Milan",
    "Roma": "AS Roma",
    "Inter": "Inter Milan",
    "Celta": "Celta de Vigo", "Celta Vigo": "Celta de Vigo",
    "La-Liga": "La Liga", "LaLiga": "La Liga",
}


def normalizar_equipos(df, cols=("Home", "Away")):
    """Aplica el diccionario de alias sobre las columnas de equipo indicadas."""
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].replace(ALIAS)
    return df


def verificar_alias_conflictos():
    """Detecta pares A->B / B->A en ALIAS que se cancelan entre sí al aplicarse
    (el bug real que dejó 'Milan'/'Roma'/'Inter' sin normalizar). Se corre
    automáticamente en cada ingesta para no repetirlo."""
    conflictos = [(k, v) for k, v in ALIAS.items() if v in ALIAS and ALIAS[v] == k]
    if conflictos:
        raise ValueError(
            f"ALIAS tiene entradas bidireccionales que se anulan entre sí: {conflictos}. "
            "Elimina la entrada vieja/incorrecta antes de continuar."
        )
    return True


def agregar_alias_permanente(nuevos_alias, path=None):
    """
    Agrega entradas al diccionario ALIAS de forma permanente: las escribe
    directamente en este archivo .py (justo antes del cierre del diccionario)
    y las aplica también en memoria para la sesión actual. Así, si una ingesta
    detecta equipos sin matchear, se resuelve con una llamada — sin editar el
    .py a mano.

    nuevos_alias : dict, ej. {"Bayer 04 Leverkusen": "Bayer Leverkusen"}
    path         : por defecto, este mismo archivo (__file__).
    """
    path = path or __file__
    conflictos = [(k, v) for k, v in nuevos_alias.items() if ALIAS.get(v) == k]
    if conflictos:
        raise ValueError(
            f"Estas entradas crearían un conflicto bidireccional con ALIAS existente: {conflictos}. "
            "No se escribió nada."
        )

    with open(path, "r", encoding="utf-8") as f:
        contenido = f.read()

    ancla = '    "La-Liga": "La Liga", "LaLiga": "La Liga",\n}'
    if ancla not in contenido:
        raise RuntimeError("No encontré el ancla esperada al final de ALIAS; agregá las entradas a mano.")

    lineas_nuevas = "\n".join(f'    "{k}": "{v}",' for k, v in nuevos_alias.items())
    reemplazo = ancla.replace("\n}", f"\n{lineas_nuevas}\n}}")
    contenido = contenido.replace(ancla, reemplazo, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(contenido)

    ALIAS.update(nuevos_alias)
    print(f"Agregados {len(nuevos_alias)} alias a {path} y aplicados en esta sesión:")
    for k, v in nuevos_alias.items():
        print(f'   "{k}" -> "{v}"')
