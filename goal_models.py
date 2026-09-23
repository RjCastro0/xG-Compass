"""
=============================================================================
 GOAL MODELS — los dos motores de predicción de goles
=============================================================================
Dos modelos con roles distintos, cada uno validado por separado:
  - TimeDecayGoalModel: histórico puro (solo goles) -> insumo de la
    dificultad de fixtures (`prior_model` en la app).
  - HybridGoalModel: compuesto de la temporada actual + corrección
    Dixon-Coles -> predicción de partidos y Monte Carlo de temporada
    (64.1% de acierto walk-forward, vs 53.8% del prior solo-goles).
"""
from math import exp, factorial

import numpy as np
import pandas as pd

from config import HALF_LIFE_DAYS, SHRINKAGE_STRENGTH


class TimeDecayGoalModel:
    """
    Estima 4 ratings por equipo (1.0 = promedio de liga):
      home_attack / away_attack   -> goles anotados, en casa / fuera (mayor = mejor)
      home_defense / away_defense -> goles concedidos, en casa / fuera (menor = mejor)

    Los partidos recientes pesan más (decaimiento exponencial con half_life_days),
    y equipos con poco historial se acercan al promedio de liga (shrinkage_strength),
    para que un recién ascendido no arranque con un rating extremo.
    """

    def __init__(self, half_life_days=730, shrinkage_strength=10.0):
        self.half_life_days = half_life_days
        self.shrinkage_strength = shrinkage_strength

    def _decay_factor(self, elapsed_days):
        return 0.5 ** (elapsed_days / self.half_life_days)

    def fit(self, X, y=None):
        self.league_home_goals_ = X["home_goals"].mean()
        self.league_away_goals_ = X["away_goals"].mean()

        team_state = {}
        for match in X.itertuples(index=False):
            for team, role, scored, conceded in (
                (match.Home, "home", match.home_goals, match.away_goals),
                (match.Away, "away", match.away_goals, match.home_goals),
            ):
                s = team_state.get(team)
                if s is None:
                    s = {
                        "home_gf": 0.0, "home_ga": 0.0, "home_weight": 0.0,
                        "away_gf": 0.0, "away_ga": 0.0, "away_weight": 0.0,
                        "last_date": match.Date,
                    }
                    team_state[team] = s

                decay = self._decay_factor((match.Date - s["last_date"]).days)
                for key in ["home_gf", "home_ga", "home_weight",
                            "away_gf", "away_ga", "away_weight"]:
                    s[key] *= decay

                s[f"{role}_gf"] += scored
                s[f"{role}_ga"] += conceded
                s[f"{role}_weight"] += 1.0
                s["last_date"] = match.Date

        self.team_state_ = team_state
        return self

    def get_ratings(self, team, reference_date):
        """(home_attack, home_defense, away_attack, away_defense) a una fecha dada."""
        k = self.shrinkage_strength
        s = self.team_state_.get(team)
        if s is None:
            return 1.0, 1.0, 1.0, 1.0

        decay = self._decay_factor((reference_date - s["last_date"]).days)
        hw, aw = s["home_weight"] * decay, s["away_weight"] * decay
        h_gf, h_ga = s["home_gf"] * decay, s["home_ga"] * decay
        a_gf, a_ga = s["away_gf"] * decay, s["away_ga"] * decay

        home_attack  = (h_gf + k * self.league_home_goals_) / (hw + k) / self.league_home_goals_
        home_defense = (h_ga + k * self.league_away_goals_) / (hw + k) / self.league_away_goals_
        away_attack  = (a_gf + k * self.league_away_goals_) / (aw + k) / self.league_away_goals_
        away_defense = (a_ga + k * self.league_home_goals_) / (aw + k) / self.league_home_goals_
        return home_attack, home_defense, away_attack, away_defense

    def expected_goals(self, home_team, away_team, reference_date):
        """Goles esperados (lambda_home, lambda_away) para home_team vs away_team."""
        home_attack, _, _, _ = self.get_ratings(home_team, reference_date)
        _, _, _, away_defense = self.get_ratings(away_team, reference_date)
        _, home_defense, _, _ = self.get_ratings(home_team, reference_date)
        away_attack = self.get_ratings(away_team, reference_date)[2]

        lambda_home = home_attack * away_defense * self.league_home_goals_
        lambda_away = away_attack * home_defense * self.league_away_goals_
        return lambda_home, lambda_away


def _poisson_pmf(k, lam):
    """P(X=k) para X ~ Poisson(lam). Implementación propia: evita depender de scipy."""
    return exp(-lam) * lam**k / factorial(k)


def _tau(x, y, lam, mu, rho):
    """Corrección de Dixon-Coles para marcadores bajos (0-0, 1-0, 0-1, 1-1)."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


# --- pesos del compuesto de ataque/defensa, derivados de correlación medida
# contra goles futuros reales (La Liga 26-27, ver validación en el chat) ---
PESOS_ATAQUE = {
    "xg": 0.176, "ocasiones_claras": 0.172, "xa": 0.162, "goles": 0.151,
    "tiros_puerta": 0.141, "tiros": 0.138, "pct_puerta": 0.061,
}
PESOS_DEFENSA = {
    "xga": 0.35, "goles_c": 0.25, "tiros_puerta_c": 0.20,
    "ppda": 0.10, "conversion_permitida": 0.10,
}
HALF_LIFE_PARTIDOS = 7.0
K_BLEND_ATAQUE = 3.0
K_BLEND_DEFENSA = 30.0


class HybridGoalModel:
    """
    Motor de predicción final: blend de un prior histórico (TimeDecayGoalModel,
    solo goles, largo plazo) con un compuesto multi-métrica de la temporada
    actual (xG, ocasiones claras, xA, goles, tiros... para ataque; xGA, goles
    concedidos, PPDA... para defensa), más la corrección de Dixon-Coles para
    marcadores bajos.

    Validado (walk-forward, La Liga 26-27):
      v1 solo goles:                    53.8% acierto
      v2 este compuesto, sin tau:       59.0% acierto
      v2 + corrección tau (rho ajustado): 64.1% acierto  <- el que se usa acá
    """

    def __init__(self, half_life_partidos=HALF_LIFE_PARTIDOS,
                 k_blend_ataque=K_BLEND_ATAQUE, k_blend_defensa=K_BLEND_DEFENSA,
                 prior_half_life_days=HALF_LIFE_DAYS, prior_shrinkage=SHRINKAGE_STRENGTH):
        self.half_life_partidos = half_life_partidos
        self.k_blend_ataque = k_blend_ataque
        self.k_blend_defensa = k_blend_defensa
        self.prior = TimeDecayGoalModel(prior_half_life_days, prior_shrinkage)
        self.rho_ = 0.0

    @staticmethod
    def _tabla_larga(partidos_df, liga=None):
        p = partidos_df.copy()
        if liga:
            p = p[p.liga == liga]
        p["fecha"] = pd.to_datetime(p["fecha"])

        filas = []
        for _, r in p.iterrows():
            for side, opp in [("local", "visitante"), ("visitante", "local")]:
                g = lambda m, s: r.get(f"{m}_{s}")
                tiros, tiros_opp = g("tiros", side), g("tiros", opp)
                acciones_def = ((g("tackles", side) or 0) + (g("intercepciones", side) or 0)
                                + (g("faltas", side) or 0))
                filas.append({
                    "fecha": r["fecha"], "equipo": r[f"equipo_{side}"],
                    "goles": g("goles", side), "xg": g("xg", side),
                    "tiros": tiros, "tiros_puerta": g("tiros_puerta", side),
                    "pct_puerta": (g("tiros_puerta", side) / tiros) if tiros else np.nan,
                    "ocasiones_claras": g("ocasiones_claras", side), "xa": g("xa", side),
                    "goles_c": g("goles", opp), "xga": g("xg", opp),
                    "tiros_puerta_c": g("tiros_puerta", opp),
                    "conversion_permitida": (g("goles", opp) / g("tiros_puerta", opp))
                                            if g("tiros_puerta", opp) else np.nan,
                    "ppda": (g("pases_campo_propio", opp) or 0) / max(1, acciones_def),
                })
        return pd.DataFrame(filas)

    def fit(self, matches_df, partidos_df, liga="La Liga"):
        self.prior.fit(matches_df)
        larga = self._tabla_larga(partidos_df, liga=liga).sort_values("fecha").reset_index(drop=True)
        self.larga_ = larga
        metricas = set(PESOS_ATAQUE) | set(PESOS_DEFENSA)
        self.liga_medias_ = {m: larga[m].mean() for m in metricas if m in larga.columns}
        return self

    def _composite_temporada(self, team):
        obs = self.larga_[self.larga_.equipo == team]
        n = len(obs)
        if n == 0:
            return 1.0, 1.0, 0

        antiguedad = np.arange(n - 1, -1, -1)
        pesos = 0.5 ** (antiguedad / self.half_life_partidos)
        pesos /= pesos.sum()

        def ratio(metrica):
            media = self.liga_medias_.get(metrica)
            valores = obs[metrica].values
            if media is None or media == 0 or np.all(pd.isna(valores)):
                return 1.0
            valores = np.where(pd.isna(valores), media, valores)
            return float(np.dot(pesos, valores)) / media

        atq = sum(w * ratio(m) for m, w in PESOS_ATAQUE.items())
        dfn = sum(w * ratio(m) for m, w in PESOS_DEFENSA.items())
        return atq, dfn, n

    def get_ratings(self, team, reference_date):
        """(home_attack, home_defense, away_attack, away_defense) — prior + temporada."""
        ha_p, hd_p, aa_p, ad_p = self.prior.get_ratings(team, reference_date)
        atq, dfn, n = self._composite_temporada(team)

        w_atq = n / (n + self.k_blend_ataque)
        w_def = n / (n + self.k_blend_defensa)

        home_attack  = w_atq * atq + (1 - w_atq) * ha_p
        away_attack  = w_atq * atq + (1 - w_atq) * aa_p
        home_defense = w_def * dfn + (1 - w_def) * hd_p
        away_defense = w_def * dfn + (1 - w_def) * ad_p
        return home_attack, home_defense, away_attack, away_defense

    def expected_goals(self, home_team, away_team, reference_date):
        home_attack, home_defense, _, _ = self.get_ratings(home_team, reference_date)
        _, _, away_attack, away_defense = self.get_ratings(away_team, reference_date)
        lambda_home = home_attack * away_defense * self.prior.league_home_goals_
        lambda_away = away_attack * home_defense * self.prior.league_away_goals_
        return lambda_home, lambda_away

    def fit_rho(self, partidos_df, liga="La Liga", half_life_days=200, rho_grid=None):
        """
        Ajusta rho (corrección Dixon-Coles) por grid search, maximizando la
        log-verosimilitud ponderada por recencia sobre los GOLES REALES ya
        jugados esta temporada. Se llama una vez, después de fit().
        """
        p = partidos_df[partidos_df.liga == liga].copy()
        p["fecha"] = pd.to_datetime(p["fecha"])
        rho_grid = rho_grid if rho_grid is not None else np.arange(-0.30, 0.31, 0.02)

        ahora = p.fecha.max()
        pesos_partido = 0.5 ** ((ahora - p.fecha).dt.days / half_life_days)

        lambdas = []
        for _, r in p.iterrows():
            lh, la = self.expected_goals(r.equipo_local, r.equipo_visitante, r.fecha)
            lambdas.append((lh, la, int(r.goles_local), int(r.goles_visitante)))

        mejor_rho, mejor_ll = 0.0, -np.inf
        for rho in rho_grid:
            ll = 0.0
            for (lh, la, gl, gv), w in zip(lambdas, pesos_partido):
                prob = max(_poisson_pmf(gl, lh) * _poisson_pmf(gv, la) * _tau(gl, gv, lh, la, rho), 1e-10)
                ll += w * np.log(prob)
            if ll > mejor_ll:
                mejor_ll, mejor_rho = ll, rho

        self.rho_ = float(mejor_rho)
        return self.rho_

    def score_matrix(self, home_team, away_team, reference_date, max_goals=10):
        """Matriz de marcador con la corrección tau aplicada y renormalizada."""
        lh, la = self.expected_goals(home_team, away_team, reference_date)
        home_probs = np.array([_poisson_pmf(k, lh) for k in range(max_goals)])
        away_probs = np.array([_poisson_pmf(k, la) for k in range(max_goals)])
        matrix = np.outer(home_probs, away_probs)
        for x in range(2):
            for y in range(2):
                matrix[x, y] *= _tau(x, y, lh, la, self.rho_)
        # con |rho| grande y lambdas altos, tau puede dar negativo en (0,1)/(1,0)
        # (ej. 1 + lam*rho con lam>1/|rho|) -> se clipea antes de renormalizar,
        # igual que ya se hace en fit_rho para el log-likelihood.
        matrix = np.clip(matrix, 0.0, None)
        matrix = matrix / matrix.sum()
        return matrix, lh, la


def outcome_probabilities(lh, la, max_goals=10):
    """1X2 (p_home, p_draw, p_away) a partir de dos lambdas de Poisson."""
    home_probs = np.array([_poisson_pmf(k, lh) for k in range(max_goals)])
    away_probs = np.array([_poisson_pmf(k, la) for k in range(max_goals)])
    matrix = np.outer(home_probs, away_probs)
    p_home = np.tril(matrix, -1).sum()
    p_draw = np.trace(matrix).sum()
    p_away = np.triu(matrix, 1).sum()
    return p_home, p_draw, p_away


def puntos_esperados(p_home, p_draw, p_away):
    """Valor esperado de puntos para (local, visitante) dado un 1X2."""
    return 3 * p_home + 1 * p_draw, 3 * p_away + 1 * p_draw
