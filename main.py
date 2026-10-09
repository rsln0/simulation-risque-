import os
import sys
import pandas as pd

from donnees import charger_prix, calculer_rendements, poids_egaux
from mesures import var_es_historique, var_es_parametrique, var_es_monte_carlo, var_noyau, stats_annuelles
from backtest import backtest_var, test_kupiec, test_christoffersen
from stress import stress_test
from validation import etude_echantillons
from graphiques import histogramme_var, graphique_backtest, graphique_stress

METHODES = ["historique", "parametrique", "monte_carlo", "ewma"]
NIVEAUX = [0.95, 0.99]

# on peut donner un autre fichier en argument : python main.py data/prix.csv
chemin = sys.argv[1] if len(sys.argv) > 1 else "data/prix.csv"
os.makedirs("resultats", exist_ok=True)

prix = charger_prix(chemin)
rendements = calculer_rendements(prix)
poids = poids_egaux(rendements)
rdt_ptf = rendements.values @ poids

print("Fichier :", chemin)
print("Actifs :", list(prix.columns))
print("Période :", prix.index[0].date(), "->", prix.index[-1].date(), "(", len(rendements), "rendements )")

# 1. VaR et ES sur toute la période
print("\nVaR et ES (en % de perte journalière)")
lignes = []
for alpha in NIVEAUX:
    var_es = {
        "historique": var_es_historique(rdt_ptf, alpha),
        "parametrique": var_es_parametrique(rendements, poids, alpha),
        "monte_carlo": var_es_monte_carlo(rendements, poids, alpha),
    }
    # la VaR à noyau (TD1) n'a pas d'ES, on l'affiche à part
    for methode, (var, es) in var_es.items():
        lignes.append([alpha, methode, 100 * var, 100 * es])
    histogramme_var(rdt_ptf, var_es, alpha)
    lignes.append([alpha, "noyau", 100 * var_noyau(rdt_ptf, alpha), None])
table_var = pd.DataFrame(lignes, columns=["alpha", "methode", "VaR (%)", "ES (%)"])
print(table_var.round(3).to_string(index=False, na_rep="-"))
table_var.to_csv("resultats/var_es.csv", index=False)

# 2. backtest avec une fenêtre glissante de 250 jours
print("\nBacktest (fenêtre de 250 jours)")
lignes = []
for alpha in NIVEAUX:
    for methode in METHODES:
        res = backtest_var(rendements, poids, methode, alpha)
        graphique_backtest(res, methode, alpha)
        T = len(res)
        N = int(res["exception"].sum())
        lr_uc, p_uc = test_kupiec(res["exception"], alpha)
        lr_ind, p_ind, lr_cc, p_cc = test_christoffersen(res["exception"], alpha)
        lignes.append([alpha, methode, T, N, 100 * N / T, 100 * (1 - alpha), p_uc, p_ind, p_cc])
colonnes = ["alpha", "methode", "T", "exceptions", "taux (%)", "attendu (%)",
            "p Kupiec", "p indep.", "p cond."]
table_bt = pd.DataFrame(lignes, columns=colonnes)
print(table_bt.round(3).to_string(index=False))
print("Un modèle est rejeté au seuil de 5% si la p-valeur est plus petite que 0.05")
table_bt.to_csv("resultats/backtest.csv", index=False)

# 3. stress testing (sur le premier actif de la liste pour le choc ciblé)
print("\nStress tests")
table_stress = stress_test(rendements, poids, nom_actif=rendements.columns[0])
print(table_stress.round(2).to_string())
table_stress.to_csv("resultats/stress.csv")
graphique_stress(table_stress)

# 4. validation in sample / out of sample (comme dans le TD1)
# on apprend sur 2019-2021 et on teste sur 2022-2025
rdt_serie = pd.Series(rdt_ptf, index=rendements.index)
apprentissage = rdt_serie["2019":"2021"]
test = rdt_serie["2022":"2025"]
courte = apprentissage.iloc[-63:]  # environ 3 mois de bourse
print("\nValidation in sample / out of sample")
for alpha in NIVEAUX:
    print("alpha =", alpha, "(attendu :", round(100 * (1 - alpha), 1), "% de dépassements)")
    print("Apprentissage sur 3 ans :")
    print(etude_echantillons(apprentissage, test, alpha).round(2).to_string(index=False))
    print("Apprentissage sur 3 mois seulement :")
    print(etude_echantillons(courte, test, alpha).round(2).to_string(index=False))

# 5. volatilité et semi-écarts-types par année (TD2)
print("\nVolatilité annualisée par année (en %)")
table_annee = pd.DataFrame(stats_annuelles(rdt_serie),
                           columns=["année", "volatilité", "semi-écart haut", "semi-écart bas"])
print(table_annee.round(2).to_string(index=False))
table_annee.to_csv("resultats/stats_annuelles.csv", index=False)
