import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


def run_soft_weighted_prediction_pipeline():
    print("🔄 Wczytywanie bazy meczów międzypaństwowych (results.csv) i danych rynkowych...")
    results = pd.read_csv('results.csv')
    results['date'] = pd.to_datetime(results['date'])

    # Wczytujemy plik z wartościami rynkowymi kadr z bazy Transfermarkt
    nat_teams = pd.read_csv('national_teams.csv')

    # Tworzymy słownik mapujący nazwę reprezentacji na jej całkowitą wartość rynkową
    # (kolumny z national_teams.csv to zazwyczaj 'name' oraz 'total_market_value')
    market_value_map = dict(zip(nat_teams['name'], nat_teams['total_market_value']))

    df = results[results['date'] >= '2010-01-01'].copy()
    df = df.sort_values('date').reset_index(drop=True)
    print(f"✅ Liczba przeanalizowanych meczów od 2010 r.: {len(df)}")

    # Przypisywanie wartości rynkowych dla gospodarza i gościa (domyślnie 10 mln, jeśli brak w słowniku)
    df['home_market_value'] = df['home_team'].map(market_value_map).fillna(10000000)
    df['away_market_value'] = df['away_team'].map(market_value_map).fillna(10000000)

    # Różnica wartości rynkowej (w milionach euro dla czytelności modelu)
    df['market_value_diff'] = (df['home_market_value'] - df['away_market_value']) / 1_000_000

    # 1. Wagi meczu i atut boiska
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

    df['match_weight'] = df['tournament'].apply(assign_match_weight)
    df['is_home_advantage'] = ((df['home_team'] == df['country']) & (df['neutral'] == False)).astype(int)

    def get_match_outcome(row):
        if row['home_score'] > row['away_score']:
            return 2  # Wygrana gospodarza
        elif row['home_score'] == row['away_score']:
            return 1  # Remis
        else:
            return 0  # Wygrana gościa

    df['target'] = df.apply(get_match_outcome, axis=1)

    # 2. Inżynieria cech: Ważona forma + Ważone bramki + Ważone H2H
    print("📈 Obliczanie zaawansowanych wskaźników (forma, bramki, H2H) z uwzględnieniem wag...")
    team_history = {}
    h2h_history = {}

    home_form, away_form = [], []
    home_goals_scored, away_goals_scored = [], []
    home_goals_conceded, away_goals_conceded = [], []
    h2h_home_weighted_score = []

    for idx, row in df.iterrows():
        h_team = row['home_team']
        a_team = row['away_team']
        h_score = row['home_score']
        a_score = row['away_score']
        m_weight = row['match_weight']

        # Ważona forma punktowa z ostatnich 5 meczów
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

        # Ważona forma bramkowa z ostatnich 5 meczów
        if h_past:
            h_goals_s = sum([item['scored'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
            h_goals_c = sum([item['conceded'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
        else:
            h_goals_s, h_goals_c = 1.2, 1.0

        if a_past:
            a_goals_s = sum([item['scored'] * item['weight'] for item in recent_a]) / sum(
                [item['weight'] for item in recent_a])
            a_goals_c = sum([item['conceded'] * item['weight'] for item in recent_a]) / sum(
                [item['weight'] for item in recent_a])
        else:
            a_goals_s, a_goals_c = 1.0, 1.2

        home_goals_scored.append(h_goals_s)
        home_goals_conceded.append(h_goals_c)
        away_goals_scored.append(a_goals_s)
        away_goals_conceded.append(a_goals_c)

        # Ważone H2H
        pair_key = tuple(sorted([h_team, a_team]))
        pair_matches = h2h_history.get(pair_key, [])
        h_h2h_val = sum([m['weight'] for m in pair_matches[-3:] if m['winner'] == h_team]) if pair_matches else 0.0
        a_h2h_val = sum([m['weight'] for m in pair_matches[-3:] if m['winner'] == a_team]) if pair_matches else 0.0
        h2h_net = h_h2h_val - a_h2h_val
        h2h_home_weighted_score.append(h2h_net)

        # Aktualizacja historii
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

        team_history[h_team].append({'pts': h_pts, 'scored': h_score, 'conceded': a_score, 'weight': m_weight})
        team_history[a_team].append({'pts': a_pts, 'scored': a_score, 'conceded': h_score, 'weight': m_weight})

        if pair_key not in h2h_history: h2h_history[pair_key] = []
        h2h_history[pair_key].append({'winner': winner, 'weight': m_weight})

    df['home_rolling_form'] = home_form
    df['away_rolling_form'] = away_form
    df['form_diff'] = df['home_rolling_form'] - df['away_rolling_form']

    df['home_goals_scored'] = home_goals_scored
    df['home_goals_conceded'] = home_goals_conceded
    df['away_goals_scored'] = away_goals_scored
    df['away_goals_conceded'] = away_goals_conceded
    df['h2h_home_weighted_score'] = h2h_home_weighted_score

    # Pełna lista cech z uwzględnieniem realnej wartości rynkowej kadr z national_teams.csv
    feature_cols = [
        'match_weight', 'is_home_advantage',
        'home_rolling_form', 'away_rolling_form', 'form_diff',
        'home_goals_scored', 'home_goals_conceded',
        'away_goals_scored', 'away_goals_conceded',
        'h2h_home_weighted_score',
        'market_value_diff'
    ]

    X = df[feature_cols]
    y = df['target']

    # Podział chronologiczny
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Soft Weighting dla remisów
    print("⚖️ Aplikowanie kontrolowanych wag (Soft Weighting dla remisów)...")
    base_weights = df['match_weight'].iloc[:split_idx].copy()
    soft_weights = np.where(y_train == 1, base_weights * 1.5, base_weights)

    # 3. Trening modelu XGBoost
    print("🤖 Trenowanie zoptymalizowanego modelu XGBoost z wartościami rynkowymi...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=150,
        max_depth=5,
        learning_rate=0.03,
        random_state=42
    )

    model.fit(X_train, y_train, sample_weight=soft_weights)

    # 4. Ewaluacja
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\n" + "=" * 50)
    print(f"✅ WYNIK MODELU Z WARTOŚCIAMI RYNKOWYMI KADR")
    print("=" * 50)
    print(f"Dokładność (Accuracy) na zbiorze testowym: {acc:.4f}")
    print("\nSzczegółowy raport klasyfikacji:")
    print(classification_report(y_test, preds,
                                target_names=['Wygrana Gościa (0)', 'Remis (1)', 'Wygrana Gospodarza (2)']))

    return model


if __name__ == '__main__':
    run_soft_weighted_prediction_pipeline()