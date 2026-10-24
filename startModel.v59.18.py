import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from collections import Counter


def run_tournament_test_pipeline():
    print("🔄 Wczytywanie bazy meczów, rankingów FIFA oraz statystyk graczy (player_stats.csv)...")
    results = pd.read_csv('results.csv')
    results['date'] = pd.to_datetime(results['date'])

    # Wczytujemy historyczny ranking FIFA
    fifa_df = pd.read_csv('fifa_ranking-2020-12-10.csv')
    fifa_df['rank_date'] = pd.to_datetime(fifa_df['rank_date'])
    fifa_df = fifa_df.sort_values('rank_date')

    # Wczytujemy współczesne wartości rynkowe z national_teams.csv (fallback po 2020)
    nat_teams = pd.read_csv('national_teams.csv')
    modern_market_map = dict(zip(nat_teams['name'], nat_teams['total_market_value']))

    # --- AGREGACJA HISTORYCZNYCH WARTOŚCI RYNKOWYCH Z PLAYER_STATS.CSV (2005-2020) ---
    print("⏳ Przetwarzanie historycznych wartości rynkowych kadr z player_stats.csv (2005-2020)...")
    player_df = pd.read_csv('player_stats.csv', usecols=['nationality', 'year', 'value'])
    player_df = player_df.dropna(subset=['nationality', 'year', 'value'])

    market_value_history = player_df.groupby(['nationality', 'year'])['value'].sum().to_dict()

    df = results[results['date'] >= '2005-01-01'].copy()
    df = df.sort_values('date').reset_index(drop=True)
    print(f"✅ Łączna liczba meczów w bazie od 2005 roku: {len(df)}")

    # Hybrydowa funkcja pobierająca wartość rynkową
    def get_hybrid_market_value(team_name, match_dt):
        yr = match_dt.year
        if 2005 <= yr <= 2020 and (team_name, yr) in market_value_history:
            return market_value_history[(team_name, yr)]
        else:
            return modern_market_map.get(team_name, 5000000.0)

    print("⏳ Mapowanie procentowych wskaźników rynkowych i inteligentnych rankingów FIFA...")

    # --- PROCENTOWY MODEL WARTOŚCI RYNKOWEJ ---
    def percentage_market_diff(h_val, a_val):
        h_mil = max(h_val, 1_000_000) / 1_000_000
        a_mil = max(a_val, 1_000_000) / 1_000_000

        if h_mil >= a_mil:
            if a_mil == 0: a_mil = 1.0
            ratio = (h_mil - a_mil) / a_mil
            sign = 1.0
        else:
            if h_mil == 0: h_mil = 1.0
            ratio = (a_mil - h_mil) / h_mil
            sign = -1.0

        if ratio < 0.5:
            return sign * (ratio * 0.2)
        elif ratio >= 1.0:
            return sign * (1.0 + (ratio - 1.0) * 0.5)
        else:
            return sign * (0.1 + (ratio - 0.5) * 1.8)

    home_market_vals, away_market_vals, adjusted_market_diffs = [], [], []
    for _, r_row in df.iterrows():
        market_h_val = get_hybrid_market_value(r_row['home_team'], r_row['date'])
        market_a_val = get_hybrid_market_value(r_row['away_team'], r_row['date'])
        home_market_vals.append(market_h_val)
        away_market_vals.append(market_a_val)
        adjusted_market_diffs.append(percentage_market_diff(market_h_val, market_a_val))

    df['home_market_value'] = home_market_vals
    df['away_market_value'] = away_market_vals
    df['market_value_diff'] = adjusted_market_diffs

    # --- HISTORICZNE DOPASOWANIE RANKINGU FIFA Z TŁUMIENIEM MAŁYCH RÓŻNIC ---
    df = df.sort_values('date').reset_index(drop=True)

    def get_historical_fifa_features(match_df, fifa_data):
        home_ranks, home_points = [], []
        away_ranks, away_points = [], []

        fifa_groups = {country: group.sort_values('rank_date') for country, group in fifa_data.groupby('country_full')}

        for _, m_row in match_df.iterrows():
            h_team_name = m_row['home_team']
            a_team_name = m_row['away_team']
            m_date = m_row['date']

            if h_team_name in fifa_groups:
                team_fifa = fifa_groups[h_team_name]
                past_ranks = team_fifa[team_fifa['rank_date'] <= m_date]
                if not past_ranks.empty:
                    latest = past_ranks.iloc[-1]
                    home_ranks.append(latest['rank'])
                    home_points.append(latest['total_points'])
                else:
                    earliest = team_fifa.iloc[0]
                    home_ranks.append(earliest['rank'])
                    home_points.append(earliest['total_points'])
            else:
                home_ranks.append(180)
                home_points.append(0.0)

            if a_team_name in fifa_groups:
                team_fifa = fifa_groups[a_team_name]
                past_ranks = team_fifa[team_fifa['rank_date'] <= m_date]
                if not past_ranks.empty:
                    latest = past_ranks.iloc[-1]
                    away_ranks.append(latest['rank'])
                    away_points.append(latest['total_points'])
                else:
                    earliest = team_fifa.iloc[0]
                    away_ranks.append(earliest['rank'])
                    away_points.append(earliest['total_points'])
            else:
                away_ranks.append(180)
                away_points.append(0.0)

        return home_ranks, home_points, away_ranks, away_points

    h_ranks, h_pts, a_ranks, a_pts = get_historical_fifa_features(df, fifa_df)

    df['home_fifa_rank'] = h_ranks
    df['home_fifa_points'] = h_pts
    df['away_fifa_rank'] = a_ranks
    df['away_fifa_points'] = a_pts

    h_ranks_arr = np.array(h_ranks, dtype=float)
    a_ranks_arr = np.array(a_ranks, dtype=float)

    rank_diff = a_ranks_arr - h_ranks_arr

    def smart_fifa_diff(rd):
        if abs(rd) <= 4:
            return 0.0
        sign = 1 if rd > 0 else -1
        return sign * np.sqrt(abs(rd)) * 2.5

    df['fifa_power_diff'] = [smart_fifa_diff(rd) for rd in rank_diff]

    def assign_match_weight(tournament):
        tourn_lower = str(tournament).lower()
        if 'world cup' in tourn_lower and 'qualification' not in tourn_lower:
            return 1.0
        elif 'euro' in tourn_lower and 'qualification' not in tourn_lower:
            return 1.0
        elif 'qualification' in tourn_lower or 'nations league' in tourn_lower:
            return 0.8
        elif 'friendly' in tourn_lower:
            return 0.2
        else:
            return 0.6

    df['tournament_weight'] = df['tournament'].apply(assign_match_weight)

    # --- INTELIGENTNA LOGIKA ATUTU WŁASNEGO BOISKA ---
    def calculate_smart_home_advantage(row_item):
        host_country = row_item['country']
        is_neutral = row_item['neutral']
        if is_neutral:
            return 0.0

        is_home_host = (row_item['home_team'] == host_country)
        is_away_host = (row_item['away_team'] == host_country)

        if not (is_home_host or is_away_host):
            return 0.0

        tourn_lower = str(row_item['tournament']).lower()
        if 'friendly' in tourn_lower:
            return 0.2
        elif 'qualification' in tourn_lower or 'nations league' in tourn_lower:
            return 0.8 if is_home_host else 0.4
        elif ('world cup' in tourn_lower or 'euro' in tourn_lower):
            return 1.0 if is_home_host else 0.7
        else:
            return 0.5

    df['smart_home_advantage'] = df.apply(calculate_smart_home_advantage, axis=1)

    def get_binary_outcome(row_item):
        if row_item['home_score'] > row_item['away_score']:
            return 1
        elif row_item['home_score'] < row_item['away_score']:
            return 0
        else:
            return -1

    df['raw_target'] = df.apply(get_binary_outcome, axis=1)

    print("📈 Obliczanie ważonych wskaźników formy, Time Decay oraz Freshness Boost...")
    team_history = {}
    h2h_history = {}

    home_form, away_form = [], []
    home_goals_scored, away_goals_scored = [], []
    home_goals_conceded, away_goals_conceded = [], []
    home_scoring_eff, away_scoring_eff = [], []
    h2h_home_weighted_score = []
    calculated_match_weights = []

    max_date = df['date'].max()

    for _, row in df.iterrows():
        h_team = row['home_team']
        a_team = row['away_team']
        h_score = row['home_score']
        a_score = row['away_score']
        m_weight = row['tournament_weight']
        match_date_item = row['date']

        age_years = (max_date - match_date_item).days / 365.25
        if age_years <= 5.0:
            time_decay = 1.0
        elif age_years <= 10.0:
            time_decay = 1.0 - (age_years - 5.0) * 0.16
        else:
            time_decay = 0.1

        effective_weight = m_weight * time_decay
        calculated_match_weights.append(effective_weight)

        recent_h, recent_a = [], []

        h_past = team_history.get(h_team, [])
        if h_past:
            recent_h = h_past[-5:]
            weighted_pts_h, weight_sum_h = 0, 0
            for idx, item in enumerate(recent_h):
                freshness_boost = 1.0 + (idx / len(recent_h)) * 1.5
                comb_w = item['weight'] * freshness_boost
                weighted_pts_h += item['pts'] * comb_w
                weight_sum_h += comb_w
            h_rolling = weighted_pts_h / weight_sum_h if weight_sum_h > 0 else 1.5
        else:
            h_rolling = 1.5

        a_past = team_history.get(a_team, [])
        if a_past:
            recent_a = a_past[-5:]
            weighted_pts_a, weight_sum_a = 0, 0
            for idx, item in enumerate(recent_a):
                freshness_boost = 1.0 + (idx / len(recent_a)) * 1.5
                comb_w = item['weight'] * freshness_boost
                weighted_pts_a += item['pts'] * comb_w
                weight_sum_a += comb_w
            a_rolling = weighted_pts_a / weight_sum_a if weight_sum_a > 0 else 1.5
        else:
            a_rolling = 1.5

        home_form.append(h_rolling)
        away_form.append(a_rolling)

        if h_past and recent_h:
            h_goals_s = sum([item['scored'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
            h_goals_c = sum([item['conceded'] * item['weight'] for item in recent_h]) / sum(
                [item['weight'] for item in recent_h])
            last_10_h = h_past[-10:]
            h_eff = sum([1 for item in last_10_h if item['scored'] > 0]) / len(last_10_h)
        else:
            h_goals_s, h_goals_c = 1.2, 1.0
            h_eff = 0.7

        if a_past and recent_a:
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

        pair_key = tuple(sorted([h_team, a_team]))
        pair_matches = h2h_history.get(pair_key, [])
        h_h2h_val = sum([m['weight'] for m in pair_matches[-3:] if m['winner'] == h_team]) if pair_matches else 0.0
        a_h2h_val = sum([m['weight'] for m in pair_matches[-3:] if m['winner'] == a_team]) if pair_matches else 0.0
        h2h_net = h_h2h_val - a_h2h_val
        h2h_home_weighted_score.append(h2h_net)

        if h_score > a_score:
            h_pts, a_pts = 3, 0
            winner_team = h_team
        elif h_score == a_score:
            h_pts, a_pts = 1, 1
            winner_team = 'draw'
        else:
            h_pts, a_pts = 0, 3
            winner_team = a_team

        if h_team not in team_history: team_history[h_team] = []
        if a_team not in team_history: team_history[a_team] = []

        team_history[h_team].append({'pts': h_pts, 'scored': h_score, 'conceded': a_score, 'weight': effective_weight})
        team_history[a_team].append({'pts': a_pts, 'scored': a_score, 'conceded': h_score, 'weight': effective_weight})

        if pair_key not in h2h_history: h2h_history[pair_key] = []
        h2h_history[pair_key].append({'winner': winner_team, 'weight': effective_weight})

    df['match_weight'] = calculated_match_weights
    df['home_rolling_form'] = home_form
    df['away_rolling_form'] = away_form
    df['form_diff'] = (df['home_rolling_form'] - df['away_rolling_form']) * 1.35
    df['home_goals_scored'] = home_goals_scored
    df['home_goals_conceded'] = home_goals_conceded
    df['away_goals_scored'] = away_goals_scored
    df['away_goals_conceded'] = away_goals_conceded
    df['home_scoring_eff'] = home_scoring_eff
    df['away_scoring_eff'] = away_scoring_eff
    df['scoring_eff_diff'] = df['home_scoring_eff'] - df['away_scoring_eff']
    df['h2h_home_weighted_score'] = h2h_home_weighted_score

    feature_cols = [
        'match_weight', 'smart_home_advantage',
        'home_rolling_form', 'away_rolling_form', 'form_diff',
        'home_goals_scored', 'home_goals_conceded',
        'away_goals_scored', 'away_goals_conceded',
        'home_scoring_eff', 'away_scoring_eff', 'scoring_eff_diff',
        'h2h_home_weighted_score',
        'market_value_diff',
        'fifa_power_diff'
    ]

    def get_binary_outcome(row_item):
        if row_item['home_score'] > row_item['away_score']:
            return 1
        elif row_item['home_score'] < row_item['away_score']:
            return 0
        else:
            return -1

    df['raw_target'] = df.apply(get_binary_outcome, axis=1)

    # ==========================================================
    # 🎯 SELEKCJA TESTOWA: OSTATnie TURNIEJE (MŚ I EURO, BEZ KWALIFIKACJI)
    # ==========================================================
    valid_df = df[df['raw_target'] != -1].copy()

    # Wybieramy mecze turniejowe (World Cup / UEFA Euro) pomijając kwalifikacje
    tourn_mask = (
        (valid_df['tournament'].str.lower().str.contains('world cup')) |
        (valid_df['tournament'].str.lower().str.contains('uefa euro'))
    ) & (~valid_df['tournament'].str.lower().str.contains('qualification'))

    # Bierzemy np. ostatnie 250-300 meczów z turniejów finałowych (co odpowiada ok. 3 MŚ i 2 Euro)
    test_df = valid_df[tourn_mask].tail(300).copy()
    test_indices = test_df.index

    # Wszystko inne (mecze ligowe, kwalifikacje, sparingi, starsze turnieje) idzie do treningu
    train_df = valid_df.drop(test_indices).copy()

    X_train = train_df[feature_cols]
    y_train = train_df['raw_target']

    X_test = test_df[feature_cols]

    print(f"🤖 Trenowanie modelu XGBoost na {len(train_df)} meczach historycznych (cała baza)...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=350,
        max_depth=6,
        learning_rate=0.015,
        subsample=0.6,
        colsample_bytree=0.7,
        random_state=42
    )

    model.fit(X_train, y_train)

    print(f"🎯 Ewaluacja na elitarnym zbiorzie testowym ({len(test_df)} meczów z finałów MŚ i Euro)...")
    probs_home_win = model.predict_proba(X_test)[:, 1]

    # ==========================================================
    # 🧠 ZSYNCHRONIZOWANA SYMULACJA POISSONA + XGBOOST
    # ==========================================================
    exact_score_hits = 0
    exact_outcome_from_poisson = 0
    predicted_scores_list = []

    h_scored_arr = test_df['home_goals_scored'].values
    a_conceded_arr = test_df['away_goals_conceded'].values
    a_scored_arr = test_df['away_goals_scored'].values
    h_conceded_arr = test_df['home_goals_conceded'].values
    h2h_arr = test_df['h2h_home_weighted_score'].values
    home_scores = test_df['home_score'].values
    away_scores = test_df['away_score'].values
    total_test = len(test_df)

    for i in range(total_test):
        p_home = probs_home_win[i]

        l_home = max(0.2, (h_scored_arr[i] * 0.6) + (a_conceded_arr[i] * 0.4))
        l_away = max(0.2, (a_scored_arr[i] * 0.6) + (h_conceded_arr[i] * 0.4))

        h2h_val = h2h_arr[i]
        if h2h_val > 0:
            l_home += 0.15
            l_away -= 0.10
        elif h2h_val < 0:
            l_home -= 0.10
            l_away += 0.15

        confidence_delta = (p_home - 0.5) * 0.8
        l_home = max(0.2, l_home + confidence_delta)
        l_away = max(0.2, l_away - confidence_delta)

        best_h, best_a = 0, 0
        max_p = -1.0
        for h_g in range(6):
            p_h = poisson.pmf(h_g, l_home)
            for a_g in range(6):
                p = p_h * poisson.pmf(a_g, l_away)
                if p > max_p:
                    max_p = p
                    best_h, best_a = h_g, a_g

        pred_score_str = f"{best_h}:{best_a}"
        predicted_scores_list.append(pred_score_str)

        actual_h = int(home_scores[i])
        actual_a = int(away_scores[i])

        if best_h == actual_h and best_a == actual_a:
            exact_score_hits += 1

        pred_res = 2 if best_h > best_a else (1 if best_h == best_a else 0)
        actual_res = 2 if actual_h > actual_a else (1 if actual_h == actual_a else 0)
        if pred_res == actual_res:
            exact_outcome_from_poisson += 1

    acc_1x2 = (exact_outcome_from_poisson / total_test) * 100
    exact_score_acc = (exact_score_hits / total_test) * 100
    score_counts = Counter(predicted_scores_list)

    # --- RAPORT KOŃCOWY ---
    print("\n" + "=" * 65)
    print("🏆 RAPORT TESTOWY: FINAŁY MŚ ORAZ EURO (OSTATNIE TURNIEJE)")
    print("=" * 65)
    print(f"📊 Przeanalizowanych meczów turniejowych: {total_test}")
    print(f"📊 Poprawne rozstrzygnięcia 1X2: **{exact_outcome_from_poisson} / {total_test}** (**{acc_1x2:.2f}%**)")
    print(f"🎯 Idealne dokładne wyniki (bramka w bramkę): **{exact_score_hits} / {total_test}** (**{exact_score_acc:.2f}%**)\n")

    print("📊 Statystyka najczęściej typowanych dokładnych wyników:")
    for sc, count in score_counts.most_common(10):
        print(f"   - Wynik {sc}: obstawiono {count} razy")
    print("=" * 65)

    return model


if __name__ == '__main__':
    run_tournament_test_pipeline()