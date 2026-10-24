import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


def run_weighted_leagues_pipeline():
    print("🔄 Wczytywanie bazy meczów (games.csv) oraz danych klubowych (clubs.csv)...")
    games = pd.read_csv('games.csv')
    clubs = pd.read_csv('clubs.csv')

    games['date'] = pd.to_datetime(games['date'])
    games = games.sort_values('date').reset_index(drop=True)

    # Czyszczenie braków w wynikach
    games = games.dropna(subset=['home_club_goals', 'away_club_goals', 'home_club_name', 'away_club_name'])
    print(f"✅ Liczba przeanalizowanych spotkań w bazie: {len(games)}")

    # 1. Mapowanie pojemności stadionów jako miara bazy infrastrukturalnej/finansowej klubu
    club_seats = dict(zip(clubs['club_id'], clubs['stadium_seats']))
    games['home_stadium_seats'] = games['home_club_id'].map(club_seats).fillna(20000)
    games['away_stadium_seats'] = games['away_club_id'].map(club_seats).fillna(20000)

    # 2. Definicja wag i prestiżu lig (League Tiers)
    league_multipliers = {
        'GB1': 1.30,  # Premier League (Anglia)
        'ES1': 1.25,  # La Liga (Hiszpania)
        'IT1': 1.20,  # Serie A (Włochy)
        'L1': 1.20,  # Bundesliga (Niemcy)
        'FR1': 1.10,  # Ligue 1 (Francja)
        'CL': 1.35,  # Liga Mistrzów
        'EL': 1.15,  # Liga Europy
        'PO1': 0.95,  # Primeira Liga (Portugalia)
        'NL1': 0.95,  # Eredivisie (Holandia)
        'TR1': 0.85,  # Süper Lig (Turcja)
        'BE1': 0.85,  # Jupiler Pro League (Belgia)
        'RU1': 0.80,  # Premier Liga (Rosja)
        'DK1': 0.75,  # Superliga (Dania)
        'PL1': 0.75,  # Ekstraklasa (Polska)
    }

    def get_league_weight(comp_id):
        return league_multipliers.get(str(comp_id), 0.70)  # Domyślna waga dla pozostałych lig

    games['league_multiplier'] = games['competition_id'].apply(get_league_weight)

    # Różnica sił przeskalowana przez wagę ligi
    games['stadium_seats_diff'] = ((games['home_stadium_seats'] - games['away_stadium_seats']) / 1000.0) * games[
        'league_multiplier']

    # 3. Waga meczu w zależności od typu rozgrywek
    def assign_match_weight(row):
        comp_type = str(row['competition_type']).lower()
        comp_id = str(row['competition_id']).lower()

        if 'national_team_competition' in comp_type:
            return 1.0
        elif 'international_cup' in comp_type or 'cl' in comp_id:
            return 1.0
        elif 'domestic_league' in comp_type:
            return row['league_multiplier'] * 0.8  # Uwzględniamy prestiż ligi w wadze spotkania
        else:
            return 0.6

    games['match_weight'] = games.apply(assign_match_weight, axis=1)
    games['is_home_advantage'] = 1  # Standardowy atut gospodarza

    # 4. Target binarny (1 = wygrana gospodarza, 0 = wygrana gościa, -1 = remis do odrzucenia w treningu)
    def get_binary_outcome(row):
        if row['home_club_goals'] > row['away_club_goals']:
            return 1
        elif row['home_club_goals'] < row['away_club_goals']:
            return 0
        else:
            return -1

    games['raw_target'] = games.apply(get_binary_outcome, axis=1)

    print("📈 Obliczanie ważonych wskaźników formy, bramek oraz skuteczności...")
    team_history = {}

    home_form, away_form = [], []
    home_goals_scored, away_goals_scored = [], []
    home_goals_conceded, away_goals_conceded = [], []
    home_scoring_eff, away_scoring_eff = [], []

    for idx, row in games.iterrows():
        h_team = row['home_club_name']
        a_team = row['away_club_name']
        h_score = row['home_club_goals']
        a_score = row['away_club_goals']
        m_weight = row['match_weight']

        # Historia gospodarza
        h_past = team_history.get(h_team, [])
        if h_past:
            recent_h = h_past[-5:]
            sum_w_pts = sum([item['pts'] * item['weight'] for item in recent_h])
            sum_w = sum([item['weight'] for item in recent_h])
            h_rolling = sum_w_pts / sum_w if sum_w > 0 else 1.5
        else:
            h_rolling = 1.5

        # Historia gościa
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

        # Statystyki bramkowe i skuteczność
        if h_past:
            h_goals_s = sum([item['scored'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
            h_goals_c = sum([item['conceded'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
            h_eff = sum([1 for item in h_past[-10:] if item['scored'] > 0]) / len(h_past[-10:])
        else:
            h_goals_s, h_goals_c = 1.2, 1.0
            h_eff = 0.7

        if a_past:
            a_goals_s = sum([item['scored'] * item['weight'] for item in recent_a]) / sum(
                [item['weight'] for item in recent_a])
            a_goals_c = sum([item['conceded'] * item['weight'] for item in recent_a]) / sum(
                [item['weight'] for item in recent_a])
            a_eff = sum([1 for item in a_past[-10:] if item['scored'] > 0]) / len(a_past[-10:])
        else:
            a_goals_s, a_goals_c = 1.0, 1.2
            a_eff = 0.6

        home_goals_scored.append(h_goals_s)
        home_goals_conceded.append(h_goals_c)
        away_goals_scored.append(a_goals_s)
        away_goals_conceded.append(a_goals_c)
        home_scoring_eff.append(h_eff)
        away_scoring_eff.append(a_eff)

        # Zapis do historii drużyn
        if h_score > a_score:
            h_pts, a_pts = 3, 0
        elif h_score == a_score:
            h_pts, a_pts = 1, 1
        else:
            h_pts, a_pts = 0, 3

        if h_team not in team_history: team_history[h_team] = []
        if a_team not in team_history: team_history[a_team] = []

        team_history[h_team].append({'pts': h_pts, 'scored': h_score, 'conceded': a_score, 'weight': m_weight})
        team_history[a_team].append({'pts': a_pts, 'scored': a_score, 'conceded': h_score, 'weight': m_weight})

    games['home_rolling_form'] = home_form
    games['away_rolling_form'] = away_form
    games['form_diff'] = games['home_rolling_form'] - games['away_rolling_form']
    games['home_goals_scored'] = home_goals_scored
    games['home_goals_conceded'] = home_goals_conceded
    games['away_goals_scored'] = away_goals_scored
    games['away_goals_conceded'] = away_goals_conceded
    games['home_scoring_eff'] = home_scoring_eff
    games['away_scoring_eff'] = away_scoring_eff
    games['scoring_eff_diff'] = games['home_scoring_eff'] - games['away_scoring_eff']

    feature_cols = [
        'match_weight', 'is_home_advantage',
        'home_rolling_form', 'away_rolling_form', 'form_diff',
        'home_goals_scored', 'home_goals_conceded',
        'away_goals_scored', 'away_goals_conceded',
        'home_scoring_eff', 'away_scoring_eff', 'scoring_eff_diff',
        'stadium_seats_diff'
    ]

    def get_match_outcome_1x2(row):
        if row['home_club_goals'] > row['away_club_goals']:
            return 2
        elif row['home_club_goals'] == row['away_club_goals']:
            return 1
        else:
            return 0

    games['target_1x2'] = games.apply(get_match_outcome_1x2, axis=1)

    X = games[feature_cols]
    y_binary = games['raw_target']

    split_idx = int(len(games) * 0.8)
    train_mask = (y_binary.iloc[:split_idx] != -1)
    X_train = X.iloc[:split_idx][train_mask]
    y_train = y_binary.iloc[:split_idx][train_mask]

    X_test = X.iloc[split_idx:]
    y_test_1x2 = games['target_1x2'].iloc[split_idx:]

    print(f"🤖 Trenowanie modelu XGBoost z uwzględnieniem wag ligowych...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=250,
        max_depth=6,
        learning_rate=0.025,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("🎯 Ewaluacja z optymalną strefą remisu 48/52...")
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
    print(f"✅ WYNIK MODELU Z WAGAMI LIGOWYMI")
    print("=" * 50)
    print(f"Liczba meczów w zbiorze testowym: {len(y_test_1x2)}")
    print(f"Dokładność (Accuracy) na zbiorze testowym: {acc:.4f}")
    print("\nSzczegółowy raport klasyfikacji:")
    print(classification_report(y_test_1x2, custom_preds,
                                target_names=['Wygrana Gościa (0)', 'Remis (1)', 'Wygrana Gospodarza (2)']))

    return model


if __name__ == '__main__':
    run_weighted_leagues_pipeline()