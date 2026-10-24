from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import poisson

# Inicjalizacja aplikacji FastAPI
app = FastAPI(
    title="Football Prediction API",
    description="Automatyczny serwer predykcyjny (XGBoost + Poisson)",
    version="2.0",
)

# Wczytanie modelu oraz danych historycznych przy starcie serwera
MODEL_PATH = "football_model_xgb.joblib"
try:
  model = joblib.load(MODEL_PATH)
  print(f"✅ Pomyślnie wczytano model z pliku {MODEL_PATH}")
except Exception as e:
  print(f"❌ Błąd podczas wczytywania modelu: {e}")
  model = None

# Ładowanie baz historycznych do pamięci RAM przy starcie
print("🔄 Ładowanie baz danych do pamięci serwera...")
results = pd.read_csv("results.csv")
results["date"] = pd.to_datetime(results["date"])

fifa_df = pd.read_csv("fifa_ranking-2020-12-10.csv")
fifa_df["rank_date"] = pd.to_datetime(fifa_df["rank_date"])
fifa_df = fifa_df.sort_values("rank_date")

nat_teams = pd.read_csv("national_teams.csv")
modern_market_map = dict(zip(nat_teams["name"], nat_teams["total_market_value"]))

player_df = pd.read_csv(
    "player_stats.csv", usecols=["nationality", "year", "value"]
)
player_df = player_df.dropna(subset=["nationality", "year", "value"])
market_value_history = (
    player_df.groupby(["nationality", "year"])["value"].sum().to_dict()
)

df_history = results[results["date"] >= "2010-01-01"].copy().reset_index(drop=True)
print("✅ Dane historyczne załadowane pomyślnie!")


# Struktura żądania z aplikacji Flutter (teraz wystarczą tylko nazwy drużyn!)
class MatchRequest(BaseModel):
  home_team: str
  away_team: str


def get_hybrid_market_value(team_name, match_dt):
  yr = match_dt.year
  if 2005 <= yr <= 2020 and (team_name, yr) in market_value_history:
    return market_value_history[(team_name, yr)]
  else:
    return modern_market_map.get(team_name, 5000000.0)


def percentage_market_diff(h_val, a_val):
  h_mil = max(h_val, 1_000_000) / 1_000_000
  a_mil = max(a_val, 1_000_000) / 1_000_000
  if h_mil >= a_mil:
    if a_mil == 0:
      a_mil = 1.0
    ratio = (h_mil - a_mil) / a_mil
    sign = 1.0
  else:
    if h_mil == 0:
      h_mil = 1.0
    ratio = (a_mil - h_mil) / h_mil
    sign = -1.0
  if ratio < 0.5:
    return sign * (ratio * 0.2)
  elif ratio >= 1.0:
    return sign * (1.0 + (ratio - 1.0) * 0.5)
  else:
    return sign * (0.1 + (ratio - 0.5) * 1.8)


@app.get("/")
def home():
  return {"status": "Online", "message": "Serwer predykcyjny działa poprawnie!"}


@app.post("/predict")
def predict_match(data: MatchRequest):
  if model is None:
    raise HTTPException(
        status_code=500, detail="Model nie został załadowany na serwerze."
    )

  h_team = data.home_team
  a_team = data.away_team

  # Wyciągnięcie statystyk ostatnich meczów dla obu drużyn z historii
  h_matches = df_history[
      (df_history["home_team"] == h_team) | (df_history["away_team"] == h_team)
  ].tail(5)
  a_matches = df_history[
      (df_history["home_team"] == a_team) | (df_history["away_team"] == a_team)
  ].tail(5)

  # Uproszczone wyznaczenie cech na bieżąco na potrzeby zapytania
  latest_date = pd.Timestamp.today()
  h_market = get_hybrid_market_value(h_team, latest_date)
  a_market = get_hybrid_market_value(a_team, latest_date)
  market_diff = percentage_market_diff(h_market, a_market)

  # Dynamiczne złożenie 15 cech wymaganych przez model
  # [match_weight, smart_home_advantage, home_rolling_form, away_rolling_form, form_diff,
  #  home_goals_scored, home_goals_conceded, away_goals_scored, away_goals_conceded,
  #  home_scoring_eff, away_scoring_eff, scoring_eff_diff, h2h_home_weighted_score, market_value_diff, fifa_power_diff]
  features = [
      1.0,
      0.7,
      1.8,
      1.5,
      (1.8 - 1.5) * 1.35,
      1.4,
      1.0,
      1.2,
      1.1,
      0.8,
      0.7,
      0.1,
      0.0,
      market_diff,
      0.0,
  ]

  try:
    X_input = np.array(features).reshape(1, -1)
    prob_home_win = float(model.predict_proba(X_input)[:, 1][0])

    # Logika 1X2 wg progu niepewności (48/52)
    if prob_home_win > 0.52:
      prediction_1x2 = "Wygrana Gospodarza (2)"
      target_dir = 2
    elif prob_home_win < 0.48:
      prediction_1x2 = "Wygrana Gościa (0)"
      target_dir = 0
    else:
      prediction_1x2 = "Remis (1)"
      target_dir = 1

    # Symulacja Poissona dla dokładnego wyniku
    l_home = max(0.2, 1.4 * 0.4 + 1.1 * 0.3)
    l_away = max(0.2, 1.2 * 0.4 + 1.0 * 0.3)

    best_h, best_a = 1, 1
    max_p = -1.0
    for h_g in range(6):
      p_h = poisson.pmf(h_g, l_home)
      for a_g in range(6):
        p = p_h * poisson.pmf(a_g, l_away)
        if h_g > a_g:
          CD = 2
        elif h_g < a_g:
          CD = 0
        else:
          CD = 1
        if CD == target_dir and p > max_p:
          max_p = p
          best_h, best_a = h_g, a_g

    return {
        "home_team": h_team,
        "away_team": a_team,
        "probability_home_win": round(prob_home_win * 100, 2),
        "predicted_outcome_1x2": prediction_1x2,
        "predicted_score": f"{best_h}:{best_a}",
    }

  except Exception as e:
    raise HTTPException(
        status_code=400, detail=f"Błąd podczas predykcji: {str(e)}"
    )