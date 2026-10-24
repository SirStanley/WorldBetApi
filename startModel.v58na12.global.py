import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from collections import Counter
from sklearn.metrics import accuracy_score


def run_binary_draw_pipeline():
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
    print(f"✅ Łączna liczba meczów w bazie: {len(df)}")

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
        h_val = get_hybrid_market_value(r_row['home_team'], r_row['date'])
        a_val = get_hybrid_market_value(r_row['away_team'], r_row['date'])
        home_market_vals.append(h_val)
        away_market_vals.append(a_val)
        adjusted_market_diffs.append(percentage_market_diff(h_val, a_val))

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
            h_team = m_row['home_team']
            a_team = m_row['away_team']
            m_date = m_row['date']

            if h_team in fifa_groups:
                team_fifa = fifa_groups[h_team]
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

            if a_team in fifa_groups:
                team_fifa = fifa_groups[a_team]
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
            return 0.1
        else:
            return 0.6

    df['tournament_weight'] = df['tournament'].apply(assign_match_weight)

    # --- INTELIGENTNA LOGIKA ATUTU WŁASNEGO BOISKA ---
    def calculate_smart_home_advantage(row):
        is_host = (row['home_team'] == row['country']) and (row['neutral'] == False)
        if not is_host:
            return 0.0

        tourn_lower = str(row['tournament']).lower()
        if 'friendly' in tourn_lower:
            return 0.1
        elif 'qualification' in tourn_lower or 'nations league' in tourn_lower:
            return 0.8
        elif ('world cup' in tourn_lower or 'euro' in tourn_lower):
            return 0.0
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
            for i, item in enumerate(recent_h):
                freshness_boost = 1.0 + (i / len(recent_h)) * 1.5
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
            for i, item in enumerate(recent_a):
                freshness_boost = 1.0 + (i / len(recent_a)) * 1.5
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

    def get_match_outcome_1x2(row_item):
        if row_item['home_score'] > row_item['away_score']:
            return 2
        elif row_item['home_score'] < row_item['away_score']:
            return 0
        else:
            return 1

    df['target_1x2'] = df.apply(get_match_outcome_1x2, axis=1)

    # --- PODZIAŁ NA ZBIÓR TRENINGOWY I TESTOWY (80% / 20% CHRONOLOGICZNIE) ---
    valid_df = df[df['raw_target'] != -1].copy()
    split_idx = int(len(valid_df) * 0.8)

    train_df = valid_df.iloc[:split_idx]
    test_df = valid_df.iloc[split_idx:].reset_index(drop=True)

    print(f"📊 Rozmiar zbioru treningowego (80%): {len(train_df)} meczów")
    print(f"📊 Rozmiar zbioru testowego (20%): {len(test_df)} meczów")

    X_train = train_df[feature_cols]
    y_train = train_df['raw_target']

    X_test = test_df[feature_cols]
    y_test_1x2 = test_df['target_1x2']

    print(f"🤖 Trenowanie modelu głównego (mózgu) XGBoost...")
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

    print("🎯 Ewaluacja 1X2 na zbiorze testowym ze strefą niepewności (48/52)...")
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

    correct_1x2_count = sum(1 for p, a in zip(custom_preds, y_test_1x2.values) if p == a)
    acc_1x2 = (correct_1x2_count / len(test_df)) * 100

    # ==========================================================
    # 🧠 ZSYNCHRONIZOWANA SYMULACJA POISSONA I TEST ZGODNOŚCI
    # ==========================================================
    exact_score_hits = 0
    consistent_matches_count = 0
    predicted_scores_list = []

    h_scored_arr = test_df['home_goals_scored'].values
    a_conceded_arr = test_df['away_goals_conceded'].values
    a_scored_arr = test_df['away_goals_scored'].values
    h_conceded_arr = test_df['home_goals_conceded'].values
    h2h_arr = test_df['h2h_home_weighted_score'].values
    home_scores = test_df['home_score'].values
    away_scores = test_df['away_score'].values

    for i in range(len(test_df)):
        p_home = probs_home_win[i]
        target_direction = custom_preds[i]  # 2 = wygrana gospodarza, 0 = gościa, 1 = remis

        l_home = max(0.2, (h_scored_arr[i] * 0.4) + (a_conceded_arr[i] * 0.3))
        l_away = max(0.2, (a_scored_arr[i] * 0.4) + (h_conceded_arr[i] * 0.3))

        h2h_val = h2h_arr[i]
        if h2h_val > 0:
            l_home += 0.15
            l_away -= 0.10
        elif h2h_val < 0:
            l_home -= 0.10
            l_away += 0.15

        if target_direction == 1:
            avg_lambda = (l_home + l_away) / 2.0
            l_home = max(0.5, avg_lambda)
            l_away = max(0.5, avg_lambda)
        else:
            dominance_power = abs(p_home - 0.5) * 2
            if p_home > 0.85:
                l_home += dominance_power * 2.6
                l_away = max(0.2, l_away - 0.6)
            elif p_home > 0.52:
                l_home += dominance_power * 2.2
                l_away = max(0.2, l_away - 0.3)
            elif p_home < 0.48:
                l_home = max(0.2, l_home - 0.3)
                l_away += dominance_power * 2.2
            elif p_home < 0.15:
                l_home = max(0.2, l_home - 0.6)
                l_away += dominance_power * 2.6
            else:
                l_home += (p_home - 0.5) * 0.5
                l_away -= (p_home - 0.5) * 0.5

        l_home = max(0.2, l_home)
        l_away = max(0.2, l_away)

        # Wyznaczanie wyniku przez Poissona (zawsze w ramach kierunku XGBoosta)
        best_h, best_a = 0, 0
        max_score_value = -1.0

        for h_g in range(6):
            p_h = poisson.pmf(h_g, l_home)
            for a_g in range(6):
                p = p_h * poisson.pmf(a_g, l_away)

                if h_g > a_g:
                    current_dir = 2
                elif h_g < a_g:
                    current_dir = 0
                else:
                    current_dir = 1

                if current_dir == target_direction:
                    score_value = p
                    if target_direction == 1:
                        if h_g == a_g:
                            if h_g == 1: score_value *= 1.5
                            elif h_g == 2: score_value *= 1.2
                    else:
                        if h_g > 0 and a_g > 0:
                            score_value *= 1.45

                    if score_value > max_score_value:
                        max_score_value = score_value
                        best_h, best_a = h_g, a_g

        if max_score_value == -1.0:
            if target_direction == 2: best_h, best_a = 2, 1
            elif target_direction == 0: best_h, best_a = 1, 2
            else: best_h, best_a = 1, 1

        predicted_scores_list.append((best_h, best_a))
        actual_h = int(home_scores[i])
        actual_a = int(away_scores[i])

        # Trafienie dokładnego wyniku
        if best_h == actual_h and best_a == actual_a:
            exact_score_hits += 1

        # TEST ZGODNOŚCI: Sprawdzamy czy wynik wyznaczony przez Poissona pokrywa się z kierunkiem XGBoosta
        poisson_res = 2 if best_h > best_a else (1 if best_h == best_a else 0)
        if poisson_res == target_direction:
            consistent_matches_count += 1

    exact_score_acc = (exact_score_hits / len(test_df)) * 100
    consistency_rate = (consistent_matches_count / len(test_df)) * 100
    score_counts = Counter([f"{h}:{a}" for h, a in predicted_scores_list])

    # --- RAPORT KOŃCOWY ---
    print("\n" + "=" * 70)
    print("🏆 KOMPLEKSOWY RAPORT SPÓJNOŚCI MODELU HYBRYDOWEGO (80/20 TEST)")
    print("=" * 70)
    print(f"📊 Rozstrzygnięcia 1X2 (XGBoost Accuracy): **{correct_1x2_count} / {len(test_df)}** (**{acc_1x2:.2f}%**)")
    print(f"🎯 Trafione dokładne wyniki (Poisson Exact Score): **{exact_score_hits} / {len(test_df)}** (**{exact_score_acc:.2f}%**)")
    print(f"🤝 **Wskaźnik Zgodności (Consistency Rate): {consistency_rate:.2f}%** ({consistent_matches_count} / {len(test_df)} meczów)\n")

    print("📊 Top 5 najczęściej typowanych dokładnych wyników:")
    for sc, count in score_counts.most_common(12):
        print(f"   - Wynik {sc}: obstawiono {count} razy")
    print("=" * 70)

    return model


if __name__ == '__main__':
    run_binary_draw_pipeline()