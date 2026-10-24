import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


def run_binary_draw_pipeline():
    print("🔄 Wczytywanie bazy meczów (results.csv) oraz danych z national_teams.csv...")
    results = pd.read_csv('results.csv')
    results['date'] = pd.to_datetime(results['date'])

    nat_teams = pd.read_csv('national_teams.csv')
    market_value_map = dict(zip(nat_teams['name'], nat_teams['total_market_value']))
    fifa_rank_map = dict(zip(nat_teams['name'], nat_teams['fifa_ranking']))

    df = results[results['date'] >= '2002-01-01'].copy()
    df = df.sort_values('date').reset_index(drop=True)
    print(f"✅ Liczba przeanalizowanych meczów: {len(df)}")

    # 1. Cechy globalne
    df['home_market_value'] = df['home_team'].map(market_value_map).fillna(10000000)
    df['away_market_value'] = df['away_team'].map(market_value_map).fillna(10000000)
    df['market_value_diff'] = (df['home_market_value'] - df['away_market_value']) / 1_000_000

    home_rank = df['home_team'].map(fifa_rank_map).fillna(150).astype(float)
    away_rank = df['away_team'].map(fifa_rank_map).fillna(150).astype(float)
    df['fifa_power_diff'] = (100.0 / np.sqrt(home_rank)) - (100.0 / np.sqrt(away_rank))

    def assign_match_weight(tournament):
        tourn_lower = str(tournament).lower()
        if 'world cup' in tourn_lower and 'qualification' not in tourn_lower:
            return 1.0
        elif 'euro' in tourn_lower and 'qualification' not in tourn_lower:
            return 1.0
        elif 'qualification' in tourn_lower or 'nations league' in tourn_lower:
            return 0.8
        elif 'friendly' in tourn_lower:
            return 0.4
        else:
            return 0.6

    df['tournament_weight'] = df['tournament'].apply(assign_match_weight)
    df['is_home_advantage'] = ((df['home_team'] == df['country']) & (df['neutral'] == False)).astype(int)

    def get_binary_outcome(row):
        if row['home_score'] > row['away_score']:
            return 1  # Wygrana gospodarza
        elif row['home_score'] < row['away_score']:
            return 0  # Wygrana gościa
        else:
            return -1  # Remis - odrzucamy z treningu binarnego

    df['raw_target'] = df.apply(get_binary_outcome, axis=1)

    print("📈 Obliczanie ważonych wskaźników formy, bramek, H2H z uwzględnieniem Time Decay...")
    team_history = {}
    h2h_history = {}

    home_form, away_form = [], []
    home_goals_scored, away_goals_scored = [], []
    home_goals_conceded, away_goals_conceded = [], []
    home_scoring_eff, away_scoring_eff = [], []
    h2h_home_weighted_score = []
    calculated_match_weights = []

    # BIEŻĄCA DATA ODNIESIENIA DLA TIME DECAY (najnowszy mecz w bazie lub dzisiaj)
    max_date = df['date'].max()

    for idx, row in df.iterrows():
        h_team = row['home_team']
        a_team = row['away_team']
        h_score = row['home_score']
        a_score = row['away_score']
        m_weight = row['tournament_weight']
        match_date = row['date']

        # --- WSPÓŁCZYNNIK WAGI HISTORYCZNEJ (TIME DECAY) ---
        # Ostatnie 5 lat = pełna waga (1.0). Starsze mecze stopniowo wygaszamy.
        age_years = (max_date - match_date).days / 365.25
        if age_years <= 5.0:
            time_decay = 1.0
        elif age_years <= 10.0:
            time_decay = 1.0 - (age_years - 5.0) * 0.16  # Liniowy spadek do 0.2
        else:
            time_decay = 0.2  # Minimalna waga dla meczy starszych niż 10 lat

        effective_weight = m_weight * time_decay
        calculated_match_weights.append(effective_weight)

        # Forma punktowa
        h_past = team_history.get(h_team, [])
        if h_past:
            recent_h = h_past[-5:]
            sum_w_pts = sum([item['pts'] * item['weight'] for item in recent_h])
            sum_w = sum([item['weight'] for item in recent_h])
            h_rolling = sum_w_pts / sum_w if sum_w > 0 else 1.5
        else:
            h_rolling = 1.5

        a_past = team_history.get(a_team, [])
        if a_past:
            recent_a = a_past[-5:]
            sum_w_pts_a = sum([item['pts'] * item['weight'] for item in recent_a])
            sum_w_a = sum([item['weight'] for item in recent_a])
            a_rolling = sum_w_pts_a / sum_w_a if sum_w_a > 0 else 1.5
        else:
            a_rolling = 1.5

        home_form.append(h_rolling)
        away_form.append(a_rolling)

        # Forma bramkowa i skuteczność
        if h_past:
            h_goals_s = sum([item['scored'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
            h_goals_c = sum([item['conceded'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
            last_10_h = h_past[-10:]
            h_eff = sum([1 for item in last_10_h if item['scored'] > 0]) / len(last_10_h)
        else:
            h_goals_s, h_goals_c = 1.2, 1.0
            h_eff = 0.7

        if a_past:
            a_goals_s = sum([item['scored'] * item['weight'] for item in recent_a]) / sum(
                [item['weight'] for item in recent_a])
            a_goals_c = sum([item['conceded'] * item['weight'] for item in recent_a]) / sum(
                [item['weight'] for item in recent_a])
            last_10_a = a_past[-10:]
            a_eff = sum([1 for item in last_10_a if item['scored'] > 0]) / len(last_10_a)
        else:
            a_goals_s, a_goals_c = 1.0, 1.2
            a_eff = 0.6

        home_goals_scored.append(h_goals_s)
        home_goals_conceded.append(h_goals_c)
        away_goals_scored.append(a_goals_s)
        away_goals_conceded.append(a_goals_c)
        home_scoring_eff.append(h_eff)
        away_scoring_eff.append(a_eff)

        # H2H
        pair_key = tuple(sorted([h_team, a_team]))
        pair_matches = h2h_history.get(pair_key, [])
        h_h2h_val = sum([m['weight'] for m in pair_matches[-3:] if m['winner'] == h_team]) if pair_matches else 0.0
        a_h2h_val = sum([m['weight'] for m in pair_matches[-3:] if m['winner'] == a_team]) if pair_matches else 0.0
        h2h_net = h_h2h_val - a_h2h_val
        h2h_home_weighted_score.append(h2h_net)

        # Historia
        if h_score > a_score:
            h_pts, a_pts = 3, 0
            winner = h_team
        elif h_score == a_score:
            h_pts, a_pts = 1, 1
            winner = 'draw'
        else:
            h_pts, a_pts = 0, 3
            winner = a_team

        if h_team not in team_history: team_history[h_team] = []
        if a_team not in team_history: team_history[a_team] = []

        team_history[h_team].append({'pts': h_pts, 'scored': h_score, 'conceded': a_score, 'weight': effective_weight})
        team_history[a_team].append({'pts': a_pts, 'scored': a_score, 'conceded': h_score, 'weight': effective_weight})

        if pair_key not in h2h_history: h2h_history[pair_key] = []
        h2h_history[pair_key].append({'winner': winner, 'weight': effective_weight})

    df['match_weight'] = calculated_match_weights
    df['home_rolling_form'] = home_form
    df['away_rolling_form'] = away_form
    df['form_diff'] = df['home_rolling_form'] - df['away_rolling_form']
    df['home_goals_scored'] = home_goals_scored
    df['home_goals_conceded'] = home_goals_conceded
    df['away_goals_scored'] = away_goals_scored
    df['away_goals_conceded'] = away_goals_conceded
    df['home_scoring_eff'] = home_scoring_eff
    df['away_scoring_eff'] = away_scoring_eff
    df['scoring_eff_diff'] = df['home_scoring_eff'] - df['away_scoring_eff']
    df['h2h_home_weighted_score'] = h2h_home_weighted_score

    feature_cols = [
        'match_weight', 'is_home_advantage',
        'home_rolling_form', 'away_rolling_form', 'form_diff',
        'home_goals_scored', 'home_goals_conceded',
        'away_goals_scored', 'away_goals_conceded',
        'home_scoring_eff', 'away_scoring_eff', 'scoring_eff_diff',
        'h2h_home_weighted_score',
        'market_value_diff',
        'fifa_power_diff'
    ]

    def get_match_outcome_1x2(row):
        if row['home_score'] > row['away_score']:
            return 2  # Gospodarz
        elif row['home_score'] == row['away_score']:
            return 1  # Remis
        else:
            return 0  # Gość

    df['target_1x2'] = df.apply(get_match_outcome_1x2, axis=1)

    X = df[feature_cols]
    y_binary = df['raw_target']

    # Podział 90/10 (lub zmieniasz wedle uznania np. na 0.85)
    split_idx = int(len(df) * 0.75)

    train_mask = (y_binary.iloc[:split_idx] != -1)
    X_train = X.iloc[:split_idx][train_mask]
    y_train = y_binary.iloc[:split_idx][train_mask]

    X_test = X.iloc[split_idx:]
    y_test_1x2 = df['target_1x2'].iloc[split_idx:]

    print(f"🤖 Trenowanie binarnego modelu XGBoost z Time Decay...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=350,
        max_depth=4,
        learning_rate=0.015,
        subsample=0.6,
        colsample_bytree=0.7,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("🎯 Ewaluacja z logiką strefy 50/50 dla remisów...")
    probs_home_win = model.predict_proba(X_test)[:, 1]

    lower_threshold = 0.48
    upper_threshold = 0.52

    custom_preds = []
    for p in probs_home_win:
        if p > upper_threshold:
            custom_preds.append(2)
        elif p < lower_threshold:
            custom_preds.append(0)
        else:
            custom_preds.append(1)

    acc = accuracy_score(y_test_1x2, custom_preds)

    print("\n" + "=" * 50)
    print(f"✅ WYNIK MODELU Z TIME DECAY I STREFĄ REMISU")
    print("=" * 50)
    print(f"Dokładność (Accuracy) na zbiorze testowym: {acc:.4f}")
    print("\nSzczegółowy raport klasyfikacji:")
    print(classification_report(y_test_1x2, custom_preds,
                                target_names=['Wygrana Gościa (0)', 'Remis (1)', 'Wygrana Gospodarza (2)']))

    return model


if __name__ == '__main__':
    run_binary_draw_pipeline()