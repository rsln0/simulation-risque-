import os
import matplotlib
matplotlib.use("Agg")  # pour sauvegarder les images sans ouvrir de fenêtre
import matplotlib.pyplot as plt


def histogramme_var(rdt_ptf, var_es, alpha, dossier="figures"):
    # distribution des rendements du portefeuille avec les VaR et ES
    os.makedirs(dossier, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.hist(100 * rdt_ptf, bins=60, color="lightsteelblue", edgecolor="white")
    couleurs = {"historique": "tab:blue", "parametrique": "tab:orange", "monte_carlo": "tab:green"}
    for methode, (var, es) in var_es.items():
        plt.axvline(-100 * var, color=couleurs[methode], linestyle="--", label="VaR " + methode)
    plt.title("Rendements journaliers du portefeuille et VaR à " + str(int(100 * alpha)) + "%")
    plt.xlabel("Rendement journalier (%)")
    plt.ylabel("Nombre de jours")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier + "/histogramme_var_" + str(int(100 * alpha)) + ".png", dpi=120)
    plt.close()


def graphique_backtest(resultat, methode, alpha, dossier="figures"):
    # pertes réelles, VaR prévue et jours d'exception
    os.makedirs(dossier, exist_ok=True)
    plt.figure(figsize=(11, 5))
    plt.plot(resultat.index, 100 * resultat["perte"], color="lightgray", linewidth=0.8, label="Perte du jour")
    plt.plot(resultat.index, 100 * resultat["var"], color="tab:red", linewidth=1.2, label="VaR prévue")
    exc = resultat[resultat["exception"] == 1]
    plt.scatter(exc.index, 100 * exc["perte"], color="black", s=15, zorder=3, label="Exceptions")
    plt.title("Backtest VaR " + str(int(100 * alpha)) + "% - méthode " + methode)
    plt.ylabel("Perte (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dossier + "/backtest_" + methode + "_" + str(int(100 * alpha)) + ".png", dpi=120)
    plt.close()


def graphique_stress(resultat, dossier="figures"):
    os.makedirs(dossier, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.barh(resultat.index, resultat["Perte (%)"], color="indianred")
    plt.gca().invert_yaxis()
    plt.title("Pertes du portefeuille selon les scénarios de stress")
    plt.xlabel("Perte (%)")
    plt.tight_layout()
    plt.savefig(dossier + "/stress_test.png", dpi=120)
    plt.close()
