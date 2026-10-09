# Portfolio Risk Engine

Projet personnel en Python (NumPy, pandas, matplotlib) pour mesurer le risque de marché d'un portefeuille multi-actifs : VaR et Expected Shortfall (historique, paramétrique, Monte Carlo), backtesting (tests de Kupiec et de Christoffersen), stress testing et quelques études de validation.

Je l'ai fait en prolongement du cours de Market Risk (ESILV), pour appliquer les notions des TD à un vrai portefeuille de plusieurs actifs.

## Données

Le portefeuille contient 6 ETF, un par classe d'actifs, téléchargés avec yfinance (prix de clôture ajustés, du 2 janvier 2019 au 30 décembre 2025, soit 1758 rendements journaliers) :

| Nom dans le code | ETF | Classe d'actifs |
|---|---|---|
| Actions_US | SPY | actions américaines |
| Actions_EU | EZU | actions zone euro |
| Actions_Emergents | EEM | actions pays émergents |
| Obligations | AGG | obligations américaines |
| Or | GLD | or |
| Immobilier | VNQ | immobilier coté américain |

Tout est en dollars. Le portefeuille est équipondéré (1/6 chacun) et on suppose une valeur de 1 000 000 $.

Le fichier `data/prix.csv` est dans le dépôt. Pour le régénérer : `pip install yfinance` puis `python telecharger_donnees.py`.

## Lancer le projet

```
pip install -r requirements.txt
python main.py
```

Les tableaux sont enregistrés dans `resultats/` et les graphiques dans `figures/`.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `donnees.py` | chargement des prix, rendements, poids |
| `mesures.py` | VaR et ES (historique, paramétrique, Monte Carlo), VaR à noyau, VaR à pondération exponentielle, statistiques annuelles |
| `backtest.py` | backtest en fenêtre glissante, tests de Kupiec et Christoffersen |
| `validation.py` | validation in sample / out of sample |
| `stress.py` | scénarios de stress |
| `graphiques.py` | histogramme, graphique de backtest, graphique de stress |
| `telecharger_donnees.py` | téléchargement des prix avec yfinance |
| `main.py` | lance tout et affiche les résultats |

## Méthodes et choix

Convention : VaR et ES sont des pertes positives, en % de la valeur du portefeuille sur 1 jour.

**Rendements.** Rendements simples journaliers, car le rendement du portefeuille est alors la somme pondérée des rendements des actifs.

**VaR historique.** Quantile d'ordre alpha des pertes passées du portefeuille. Aucune hypothèse sur la loi, mais on suppose que le passé ressemble à l'avenir.

**VaR paramétrique.** On suppose que le rendement du portefeuille suit une loi normale. On estime la moyenne et la matrice de covariance des actifs, puis `sigma_ptf = sqrt(w' Cov w)`. VaR = -mu + sigma * z(alpha), ES = -mu + sigma * phi(z) / (1 - alpha).

**VaR Monte Carlo.** On simule 20 000 jours de rendements d'actifs corrélés (loi normale multivariée avec la décomposition de Cholesky de la covariance), puis on calcule la VaR de ces scénarios comme pour l'historique. Avec la même hypothèse normale que la paramétrique, on retrouve des résultats très proches (bon test de cohérence).

**VaR à noyau gaussien** (TD1). Version lissée de l'historique : on met une petite cloche normale sur chaque perte observée, on additionne, puis on cherche le quantile par dichotomie. La largeur des cloches vient de la règle de Silverman. Je n'ai pas calculé d'ES pour cette méthode.

**VaR à pondération exponentielle (EWMA)** (TD2). Même formule que la paramétrique, mais les jours récents pèsent plus (poids lambda^âge avec lambda = 0,94, valeur classique de RiskMetrics). C'est une VaR « conditionnelle » : elle s'adapte à la volatilité du moment, contrairement aux autres qui sont « marginales ».

**Expected Shortfall.** Perte moyenne dans les cas où la perte dépasse la VaR. Elle regarde la queue de la distribution, ce que ne fait pas la VaR.

**Backtest.** Chaque jour, VaR calculée sur les 250 jours précédents, comparée à la perte réelle du jour. Un dépassement est une exception. Avec une VaR à 99%, on attend environ 1% d'exceptions.

**Test de Kupiec.** Teste si la fréquence d'exceptions est compatible avec 1 - alpha (rapport de vraisemblance, chi2 à 1 degré de liberté).

**Test de Christoffersen.** Teste si les exceptions sont indépendantes dans le temps (pas de paquets d'exceptions). Le test conditionnel combine Kupiec et l'indépendance (chi2 à 2 degrés de liberté). Les p-valeurs sont calculées à la main avec `math.erfc` et `exp`, ce qui évite scipy.

**Validation in sample / out of sample** (TD1). On estime la VaR sur 2019-2021, puis on regarde la proportion de jours qui la dépassent sur cette même période (in sample) et sur 2022-2025 (out of sample). On refait l'exercice avec seulement 3 mois d'apprentissage (63 jours) pour voir quelle méthode résiste le mieux avec peu de données.

**Statistiques annuelles** (TD2). Volatilité annualisée et semi-écarts-types (jours au-dessus / en dessous de la moyenne) pour chaque année.

**Stress tests.**
1. Baisse de 10% de tous les actifs
2. Chute de 30% d'un seul actif (effet = 30% x son poids)
3. Pire période de 10 jours de l'historique
4. Volatilités doublées (covariance x4) avec une VaR normale 99%
5. Corrélations mises à 1 (plus de diversification) avec une VaR normale 99%

## Résultats

VaR et ES à 1 jour, en % du portefeuille :

| alpha | méthode | VaR | ES |
|---|---|---|---|
| 95% | historique | 1.19 | 1.99 |
| 95% | paramétrique | 1.38 | 1.75 |
| 95% | Monte Carlo | 1.38 | 1.75 |
| 95% | noyau | 1.24 | - |
| 99% | historique | 2.09 | 3.80 |
| 99% | paramétrique | 1.98 | 2.27 |
| 99% | Monte Carlo | 1.96 | 2.30 |
| 99% | noyau | 2.16 | - |

Backtest (1508 jours testés, fenêtre de 250 jours) :

| alpha | méthode | exceptions | taux | p Kupiec | p indép. | p cond. |
|---|---|---|---|---|---|---|
| 95% | historique | 79 | 5.2% | 0.67 | 0.003 | 0.010 |
| 95% | paramétrique | 72 | 4.8% | 0.69 | 0.008 | 0.026 |
| 95% | Monte Carlo | 72 | 4.8% | 0.69 | 0.008 | 0.026 |
| 95% | EWMA | 97 | 6.4% | 0.014 | 0.58 | 0.043 |
| 99% | historique | 25 | 1.7% | 0.019 | 0.001 | 0.000 |
| 99% | paramétrique | 32 | 2.1% | 0.000 | 0.004 | 0.000 |
| 99% | Monte Carlo | 33 | 2.2% | 0.000 | 0.001 | 0.000 |
| 99% | EWMA | 38 | 2.5% | 0.000 | 0.083 | 0.000 |

Validation in sample / out of sample (proportion de jours au-dessus de la VaR, en %) :

| alpha | apprentissage | méthode | in sample | out of sample |
|---|---|---|---|---|
| 95% | 3 ans | historique | 5.03 | 5.59 |
| 95% | 3 ans | gaussienne | 3.17 | 2.79 |
| 95% | 3 ans | noyau | 4.37 | 4.49 |
| 95% | 3 mois | historique | 6.35 | 13.47 |
| 95% | 3 mois | gaussienne | 4.76 | 11.38 |
| 95% | 3 mois | noyau | 1.59 | 10.08 |
| 99% | 3 ans | historique | 1.06 | 0.40 |
| 99% | 3 ans | gaussienne | 1.19 | 0.60 |
| 99% | 3 ans | noyau | 0.93 | 0.30 |
| 99% | 3 mois | historique | 1.59 | 4.59 |
| 99% | 3 mois | gaussienne | 1.59 | 4.99 |
| 99% | 3 mois | noyau | 1.59 | 1.60 |

Volatilité annualisée par année (%) :

| année | volatilité | semi-écart haut | semi-écart bas |
|---|---|---|---|
| 2019 | 7.22 | 4.81 | 5.37 |
| 2020 | 23.99 | 14.94 | 18.70 |
| 2021 | 9.24 | 6.25 | 6.78 |
| 2022 | 16.63 | 12.31 | 11.12 |
| 2023 | 10.41 | 7.46 | 7.23 |
| 2024 | 9.68 | 6.39 | 7.25 |
| 2025 | 11.76 | 8.51 | 8.08 |

Stress tests :

| Scénario | Perte |
|---|---|
| Baisse de 10% de tous les actifs | 10.0% |
| Chute de 30% d'un actif (Actions_US) | 5.0% |
| Pire période de 10 jours observée (fin 18 mars 2020) | 22.3% |
| Volatilité x2 (VaR normale 99%) | 4.0% |
| Corrélations à 1 (VaR normale 99%) | 2.6% |

Les graphiques sont dans `figures/` (sur l'histogramme, la VaR paramétrique est cachée derrière celle de Monte Carlo car elles sont presque identiques).

## Interprétation

- À 99%, la VaR historique et la VaR normale sont proches (2.09% contre 1.98%), mais l'ES est très différente : 3.80% contre 2.27%. Les pires jours (surtout mars 2020) sont bien plus violents que ce que prévoit la loi normale. La VaR ne voit pas ça, l'ES si.
- À 95%, les trois méthodes de base passent le test de Kupiec (taux d'exceptions proche de 5%).
- À 99%, tous les modèles sont rejetés par Kupiec : 25 à 38 exceptions au lieu d'environ 15 attendues. La méthode normale fait pire que l'historique.
- Le test d'indépendance de Christoffersen rejette les trois méthodes classiques : les exceptions arrivent par paquets (février-mars 2020, 2022, avril 2025). Une VaR sur 250 jours réagit trop lentement quand la volatilité monte d'un coup.
- L'EWMA corrige ce défaut (indépendance non rejetée au seuil de 5%, p = 0.58 à 95% et 0.083 à 99%) mais elle sous-estime le niveau de risque (6.4% d'exceptions à 95%). Mémoire courte et loi normale : elle s'adapte vite, mais n'a pas assez d'information sur les queues.
- In sample, les VaR collent à ce qu'on attend, sauf qu'à 95% la gaussienne est trop prudente (3.2% au lieu de 5%). Out of sample, l'apprentissage de 2019-2021 donne à 99% une VaR trop haute (0.3% à 0.6% de dépassements) parce qu'il contient le krach de mars 2020.
- Avec 3 mois de données, toutes les méthodes sous-estiment fortement le risque (jusqu'à 13% de dépassements au lieu de 5%). La méthode à noyau s'en sort mieux à 99% (1.6%) car elle lisse la queue, mais cela reste un résultat sur une seule période de test.
- En 2020, le semi-écart bas est nettement plus grand que le haut (18.7 contre 14.9) : année très risquée à la baisse. En 2022, c'est l'inverse, avec de grosses remontées malgré l'année difficile.
- La pire période de 10 jours (22.3%) est très au-dessus de la VaR à 1 jour : la VaR dit peu de choses sur ce qui se passe en vraie crise.

## Limites et pistes

- Hypothèse de normalité pour trois méthodes sur cinq.
- Fenêtre fixe de 250 jours et poids constants.
- Horizon de 1 jour seulement.
- La période comprend le krach de 2020, ce qui pèse beaucoup sur les résultats.
- Pas de coûts de transaction, pas de rebalancement.
- Les scénarios de stress sont des chocs choisis à la main, pas calibrés sur des crises précises.
- Pistes : GARCH pour la volatilité, VaR sur plusieurs jours, VaR d'un produit non linéaire (option) par Monte Carlo comme dans le TD2.

## Questions possibles en entretien

- Pourquoi l'ES est-elle préférée à la VaR par la réglementation ? Elle tient compte de la taille des pertes au-delà de la VaR et elle est sous-additive.
- Pourquoi la VaR normale est-elle trop basse à 99% ? Les queues réelles sont plus épaisses.
- Que veut dire une p-valeur de 0.001 à Kupiec ? La fréquence d'exceptions est incompatible avec le niveau de confiance annoncé.
- Pourquoi Cholesky ? Pour simuler des actifs corrélés à partir de lois normales indépendantes.
- Pourquoi multiplier la covariance par 4 et pas par 2 ? La covariance est en volatilité au carré.
- Différence entre VaR marginale et conditionnelle ? La marginale utilise toute la période de la même façon, la conditionnelle (EWMA) donne plus de poids aux jours récents.
