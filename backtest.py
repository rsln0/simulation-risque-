import math
import numpy as np
import pandas as pd

from mesures import var_es_historique, var_es_parametrique, var_es_monte_carlo, var_es_ewma


def calculer_var(methode, passe, poids, alpha):
    # passe = rendements des actifs sur la fenêtre d'estimation
    if methode == "historique":
        rdt_ptf = passe.values @ poids
        return var_es_historique(rdt_ptf, alpha)[0]
    if methode == "parametrique":
        return var_es_parametrique(passe, poids, alpha)[0]
    if methode == "monte_carlo":
        return var_es_monte_carlo(passe, poids, alpha)[0]
    if methode == "ewma":
        return var_es_ewma(passe, poids, alpha)[0]
    raise ValueError("méthode inconnue : " + methode)


def backtest_var(rendements, poids, methode, alpha, fenetre=250):
    # chaque jour t, on calcule la VaR avec les 250 jours d'avant (sans regarder le jour t)
    # puis on regarde si la perte réelle du jour t dépasse cette VaR
    rdt_ptf = rendements.values @ poids
    var_prevues = []
    for t in range(fenetre, len(rendements)):
        passe = rendements.iloc[t - fenetre:t]
        var_prevues.append(calculer_var(methode, passe, poids, alpha))

    resultat = pd.DataFrame(index=rendements.index[fenetre:])
    resultat["perte"] = -rdt_ptf[fenetre:]
    resultat["var"] = var_prevues
    # exception = 1 si la perte du jour est plus grande que la VaR prévue
    resultat["exception"] = (resultat["perte"] > resultat["var"]).astype(int)
    return resultat


def log_vrais(k, n, p):
    # log de p^k * (1-p)^(n-k), avec un test pour éviter log(0)
    res = 0.0
    if k > 0:
        res += k * math.log(p)
    if n - k > 0:
        res += (n - k) * math.log(1 - p)
    return res


def test_kupiec(exceptions, alpha):
    # H0 : la proportion d'exceptions est égale à 1 - alpha
    e = np.asarray(exceptions)
    T = len(e)
    N = int(e.sum())
    p = 1 - alpha
    pi_obs = N / T
    lr_uc = -2 * (log_vrais(N, T, p) - log_vrais(N, T, pi_obs))
    lr_uc = max(lr_uc, 0.0)
    # la statistique suit un chi2 à 1 degré de liberté, sa p-valeur vaut erfc(sqrt(x/2))
    p_valeur = math.erfc(math.sqrt(lr_uc / 2))
    return lr_uc, p_valeur


def test_christoffersen(exceptions, alpha):
    # H0 (indépendance) : avoir une exception aujourd'hui ne change pas la proba d'en avoir demain
    e = np.asarray(exceptions)
    n00 = int(np.sum((e[:-1] == 0) & (e[1:] == 0)))
    n01 = int(np.sum((e[:-1] == 0) & (e[1:] == 1)))
    n10 = int(np.sum((e[:-1] == 1) & (e[1:] == 0)))
    n11 = int(np.sum((e[:-1] == 1) & (e[1:] == 1)))

    total = n00 + n01 + n10 + n11
    pi = (n01 + n11) / total
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    ll_h0 = log_vrais(n01 + n11, total, pi)
    ll_h1 = log_vrais(n01, n00 + n01, pi01) + log_vrais(n11, n10 + n11, pi11)
    lr_ind = max(-2 * (ll_h0 - ll_h1), 0.0)
    p_ind = math.erfc(math.sqrt(lr_ind / 2))

    # test conjoint : Kupiec + indépendance, chi2 à 2 degrés de liberté
    # pour 2 ddl la p-valeur s'écrit simplement exp(-x/2)
    lr_uc, _ = test_kupiec(e, alpha)
    lr_cc = lr_uc + lr_ind
    p_cc = math.exp(-lr_cc / 2)
    return lr_ind, p_ind, lr_cc, p_cc
