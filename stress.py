import numpy as np
import pandas as pd

from mesures import var_es_normale

VALEUR_PTF = 1_000_000  # valeur du portefeuille en dollars (hypothèse)


def stress_test(rendements, poids, nom_actif, alpha=0.99):
    # chaque scénario donne une perte en % du portefeuille
    noms = list(rendements.columns)
    mu = rendements.mean().values
    cov = rendements.cov().values
    scenarios = {}

    # 1. baisse de 10% de tous les actifs en même temps
    scenarios["Baisse de 10% de tous les actifs"] = 0.10

    # 2. gros choc sur un seul actif (-30%), l'effet dépend de son poids
    i = noms.index(nom_actif)
    scenarios["Chute de 30% de " + nom_actif] = 0.30 * poids[i]

    # 3. pire période de 10 jours de l'historique (rendement cumulé du portefeuille)
    rdt_ptf = pd.Series(rendements.values @ poids, index=rendements.index)
    cumul_10j = (1 + rdt_ptf).rolling(10).apply(np.prod, raw=True) - 1
    scenarios["Pire période de 10 jours observée"] = -cumul_10j.min()

    # 4. volatilités doublées : les écarts-types x2 donc la covariance x4
    sigma_x2 = np.sqrt(poids @ (4 * cov) @ poids)
    var_vol, _ = var_es_normale(mu @ poids, sigma_x2, alpha)
    scenarios["Volatilité x2 (VaR normale 99%)"] = var_vol

    # 5. toutes les corrélations à 1 (en crise tout baisse ensemble, plus de diversification)
    vols = np.sqrt(np.diag(cov))
    cov_corr1 = np.outer(vols, vols)
    sigma_corr1 = np.sqrt(poids @ cov_corr1 @ poids)
    var_corr1, _ = var_es_normale(mu @ poids, sigma_corr1, alpha)
    scenarios["Corrélations à 1 (VaR normale 99%)"] = var_corr1

    resultat = pd.DataFrame({
        "Perte (%)": [100 * v for v in scenarios.values()],
        "Perte ($)": [VALEUR_PTF * v for v in scenarios.values()],
    }, index=list(scenarios.keys()))
    return resultat
