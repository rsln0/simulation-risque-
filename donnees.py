import numpy as np
import pandas as pd


def charger_prix(chemin):
    # le csv a une colonne Date puis une colonne par actif (prix de clôture)
    prix = pd.read_csv(chemin, index_col="Date", parse_dates=True)
    prix = prix.sort_index()
    # les marchés n'ont pas les mêmes jours fériés, on enlève les lignes avec des trous
    prix = prix.dropna()
    return prix


def calculer_rendements(prix):
    # rendements simples journaliers : P(t) / P(t-1) - 1
    # j'ai pris les rendements simples (pas les log) car le rendement du portefeuille
    # est alors exactement la somme pondérée des rendements des actifs
    rendements = prix.pct_change().dropna()
    return rendements


def poids_egaux(rendements):
    # portefeuille équipondéré, c'est le plus simple pour commencer
    n_actifs = rendements.shape[1]
    return np.ones(n_actifs) / n_actifs
