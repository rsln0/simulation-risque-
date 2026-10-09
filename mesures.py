import numpy as np
from statistics import NormalDist

# Convention : la VaR et l'ES sont données comme des pertes positives.
# Exemple : VaR 99% = 2% veut dire "dans 99% des jours on perd moins de 2%".


def var_es_historique(rdt_ptf, alpha):
    # on passe en pertes et on prend le quantile alpha de la distribution observée
    pertes = -np.asarray(rdt_ptf)
    var = np.quantile(pertes, alpha)
    # l'ES est la perte moyenne quand on dépasse la VaR
    es = pertes[pertes >= var].mean()
    return var, es


def var_es_normale(mu, sigma, alpha):
    # formules quand le rendement du portefeuille suit une loi normale(mu, sigma)
    z = NormalDist().inv_cdf(alpha)
    var = -mu + sigma * z
    es = -mu + sigma * NormalDist().pdf(z) / (1 - alpha)
    return var, es


def var_es_parametrique(rendements, poids, alpha):
    # on estime la moyenne et la matrice de covariance des actifs
    # puis on calcule la moyenne et l'écart-type du portefeuille
    mu_actifs = rendements.mean().values
    cov = rendements.cov().values
    mu_ptf = mu_actifs @ poids
    sigma_ptf = np.sqrt(poids @ cov @ poids)
    return var_es_normale(mu_ptf, sigma_ptf, alpha)


def var_es_monte_carlo(rendements, poids, alpha, n_sim=20000, graine=0):
    # on simule des rendements d'actifs corrélés avec la décomposition de Cholesky
    # si cov = L L^T alors mu + L z (avec z normale standard) a bien la covariance cov
    rng = np.random.default_rng(graine)
    mu_actifs = rendements.mean().values
    cov = rendements.cov().values
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((n_sim, len(mu_actifs)))
    rdt_simules = mu_actifs + z @ L.T
    # une fois les scénarios simulés, on fait comme pour la méthode historique
    rdt_ptf_simules = rdt_simules @ poids
    return var_es_historique(rdt_ptf_simules, alpha)


def var_noyau(rdt_ptf, alpha):
    # VaR non paramétrique avec un noyau gaussien (comme dans le TD1)
    # idée : au lieu de la fonction de répartition en escalier de l'historique,
    # on met une petite cloche normale sur chaque perte observée et on les additionne
    pertes = -np.asarray(rdt_ptf)
    n = len(pertes)
    loi = NormalDist()
    # largeur de la cloche : règle de Silverman
    h = 1.06 * pertes.std() * n ** (-1 / 5)

    def repartition(x):
        return np.mean([loi.cdf((x - p) / h) for p in pertes])

    # on cherche x tel que repartition(x) = alpha, par dichotomie
    bas = pertes.min() - 5 * h
    haut = pertes.max() + 5 * h
    for _ in range(50):
        milieu = (bas + haut) / 2
        if repartition(milieu) < alpha:
            bas = milieu
        else:
            haut = milieu
    return (bas + haut) / 2


def var_es_ewma(rendements, poids, alpha, lam=0.94):
    # même formule que la méthode paramétrique mais les jours récents comptent plus
    # poids du jour i (le plus récent a le poids le plus fort) : lam^(âge)
    # lam = 0.94 est la valeur classique de RiskMetrics pour des données journalières
    valeurs = rendements.values
    n = len(valeurs)
    w = lam ** np.arange(n)[::-1]
    w = w / w.sum()
    mu_actifs = np.average(valeurs, axis=0, weights=w)
    cov = np.cov(valeurs.T, aweights=w)
    mu_ptf = mu_actifs @ poids
    sigma_ptf = np.sqrt(poids @ cov @ poids)
    return var_es_normale(mu_ptf, sigma_ptf, alpha)


def stats_annuelles(rdt_ptf):
    # volatilité et semi-écarts-types (hausse et baisse) par année, annualisés (TD2 question 6)
    # semi-écart-type bas = on ne regarde que les jours en dessous de la moyenne
    lignes = []
    for annee in sorted(set(rdt_ptf.index.year)):
        r = rdt_ptf[rdt_ptf.index.year == annee]
        m = r.mean()
        vol = r.std()
        haut = np.sqrt(np.mean(np.maximum(r - m, 0) ** 2))
        bas = np.sqrt(np.mean(np.maximum(m - r, 0) ** 2))
        coef = np.sqrt(252)
        lignes.append([annee, 100 * vol * coef, 100 * haut * coef, 100 * bas * coef])
    return lignes
