import numpy as np
import pandas as pd

from mesures import var_es_historique, var_es_normale, var_noyau


def calcul_var(methode, rdt_ptf, alpha):
    # rdt_ptf = rendements du portefeuille (une seule série) sur la période d'apprentissage
    if methode == "historique":
        return var_es_historique(rdt_ptf, alpha)[0]
    if methode == "gaussienne":
        mu = np.mean(rdt_ptf)
        sigma = np.std(rdt_ptf, ddof=1)
        return var_es_normale(mu, sigma, alpha)[0]
    if methode == "noyau":
        return var_noyau(rdt_ptf, alpha)
    raise ValueError("méthode inconnue : " + methode)


def proportion_depassements(rdt_ptf, var):
    # part des jours où la perte est plus grande que la VaR
    return np.mean(-np.asarray(rdt_ptf) > var)


def etude_echantillons(apprentissage, test, alpha):
    # on estime la VaR sur la période d'apprentissage puis on regarde
    # - in sample : la proportion de dépassements sur cette même période
    # - out of sample : la proportion de dépassements sur la période de test (TD1 question 2)
    lignes = []
    for methode in ["historique", "gaussienne", "noyau"]:
        var = calcul_var(methode, apprentissage.values, alpha)
        dans = proportion_depassements(apprentissage.values, var)
        hors = proportion_depassements(test.values, var)
        lignes.append([methode, 100 * var, 100 * dans, 100 * hors])
    return pd.DataFrame(lignes, columns=["methode", "VaR (%)", "in sample (%)", "out of sample (%)"])
