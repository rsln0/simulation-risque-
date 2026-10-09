import os
import yfinance as yf  # pip install yfinance (seulement pour ce script)

# Télécharge de vrais prix d'ETF sur Yahoo Finance.
# Un ETF par classe d'actifs pour avoir un portefeuille diversifié.
etfs = {
    "SPY": "Actions_US",
    "EZU": "Actions_EU",
    "EEM": "Actions_Emergents",
    "AGG": "Obligations",
    "GLD": "Or",
    "VNQ": "Immobilier",
}

prix = yf.download(list(etfs.keys()), start="2019-01-01", end="2025-12-31", auto_adjust=True)["Close"]
prix = prix[list(etfs.keys())].rename(columns=etfs)
prix.index.name = "Date"
prix = prix.dropna()

os.makedirs("data", exist_ok=True)
prix.to_csv("data/prix.csv")
print("Fichier data/prix.csv créé :", prix.shape)
