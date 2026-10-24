import os
import pandas as pd
import numpy as np
from scipy.stats import poisson


def run_advanced_form_h2h_power_ranking():
    results_file = 'results.csv'
    fifa_file = 'fifa_ranking-2020-12-10.csv'
    players_file = 'player_stats.csv'

    # 1. Sprawdzenie plików
    for f in [results_file, fifa_file]:
        if not os.path.exists(f):
            print(f"❌ Brak wymaganego pliku: '{f}'!")
            return

    print("📥 Wczytywanie danych z uwzględnieniem formy i H2H...")
    results_df = pd.read_csv(results_file)
    results_df['date'] = pd.to_datetime(results_df['date'])

    fifa_df = pd.read_csv(fifa_file)
    fifa_df['rank_date'] = pd.to_datetime(fifa_df['rank_date'])
    latest_fifa_date = fifa_df['rank_date'].max()

    latest_fifa_subset = fifa_df[fifa_df['rank_date'] == latest_fifa_date].copy()

    # Odrzucamy wykluczone drużyny (np. Rosja)
    excluded_teams = ['Russia']
    latest_fifa_subset = latest_fifa_subset[~latest_fifa_subset['country_full'].isin(excluded_teams)]

    top_50_df = latest_fifa_subset.sort_values('rank').head(50)
    top_50 = top_50_df['country_full'].tolist()
    team_to_rank = dict(zip(top_50_df['country_full'], top_50_df['rank']))

    print(f"✅ Pobrano elitarne Top 50 drużyn.")

    # 2. Skokowe wagi rynkowe (Market Value Tiers)
    team_market_tiers = {}
    if os.path.exists(players_file):
        players_df = pd.read_csv(players_file, low_memory=False)
        val_col = 'market_value' if 'market_value' in players_df.columns else 'value_eur'
        team_col = 'team' if 'team' in players_df.columns else (
            'country' if 'country' in players_df.columns else 'nationality')

        if val_col in players_df.columns and team_col in players_df.columns:
            recent_players = players_df[players_df.get('year', 2020) >= 2018]
            team_total_val = recent_players.groupby(team_col)[val_col].sum().to_dict()

            for team, val in team_total_val.items():
                if val >= 1_000_000_000:
                    team_market_tiers[team] = 1.6
                elif val >= 500_000_000:
                    team_market_tiers[team] = 1.40
                elif val >= 200_000_000:
                    team_market_tiers[team] = 1.15
                elif val >= 100_000_000:
                    team_market_tiers[team] = 1.00
                else:
                    team_market_tiers[team] = 0.95

    # 3. Wagi turniejowe i budowanie historii meczów
    def get_tournament_weight(tournament):
        t = str(tournament).lower()
        if any(x in t for x in ['world cup', 'euro', 'copa america', 'confederations', 'asian cup', 'afcon']):
            return 1.0
        elif any(x in t for x in ['nations league', 'qualification', 'WC qualification', 'Euro qualification']):
            return 0.6
        else:
            return 0.2

    cutoff_date = pd.to_datetime('2018-01-01')
    results_df = results_df[results_df['date'] >= cutoff_date]
    results_sorted = results_df.sort_values('date')

    team_form = {}
    h2h_database = {}

    for _, row in results_sorted.iterrows():
        h, a = row['home_team'], row['away_team']
        h_score, a_score = row['home_score'], row['away_score']
        w = get_tournament_weight(row['tournament'])

        if pd.isna(h_score) or pd.isna(a_score): continue

        # Zapis formy ogólnej
        if h not in team_form: team_form[h] = []
        team_form[h].append({'scored': h_score, 'conceded': a_score, 'weight': w})

        if a not in team_form: team_form[a] = []
        team_form[a].append({'scored': a_score, 'conceded': h_score, 'weight': w})

        # Zapis bezpośrednich pojedynków (H2H) dla pary (team1, team2) niezależnie od gospodarza
        pair_key = tuple(sorted([h, a]))
        if pair_key not in h2h_database: h2h_database[pair_key] = []
        h2h_database[pair_key].append({
            'home': h, 'away': a, 'home_score': h_score, 'away_score': a_score
        })

    # Wstępne obliczenie bazowego potencjału z ostatnich 10 meczów
    team_power_index = {}
    for team in top_50:
        history = team_form.get(team, [])
        # Bierzemy OSTATNIE 10 meczów dla aktualnej formy
        recent_history = history[-10:] if len(history) >= 10 else history

        if recent_history:
            total_w = sum([x['weight'] for x in recent_history])
            w_scored = sum([x['scored'] * x['weight'] for x in recent_history]) / total_w if total_w > 0 else 1.2
            w_conceded = sum([x['conceded'] * x['weight'] for x in recent_history]) / total_w if total_w > 0 else 1.0
        else:
            w_scored, w_conceded = 1.2, 1.0

        market_multiplier = team_market_tiers.get(team, 1.0)

        team_power_index[team] = {
            'attack': w_scored * market_multiplier,
            'defense': w_conceded / market_multiplier
        }

    # 4. Uruchomienie symulacji Monte Carlo (50 turniejów) z uwzględnieniem H2H (ostatnie 3 mecze)
    num_simulations = 50
    print(f"\n🏆 Uruchamianie {num_simulations} symulacji z uwzględnieniem formy (10 meczów) oraz H2H (3 mecze)...")
    print("   (To chwilkę potrwa. Obliczamy dynamikę bezpośrednich starć...)\n")

    total_simulation_points = {team: 0 for team in top_50}
    podium_counts = {team: 0 for team in top_50}
    win_counts = {team: 0 for team in top_50}

    for sim in range(num_simulations):
        sim_points = {team: 0 for team in top_50}

        for i in range(len(top_50)):
            for j in range(len(top_50)):
                if i == j: continue
                home, away = top_50[i], top_50[j]

                h_stats = team_power_index.get(home, {'attack': 1.2, 'defense': 1.0})
                a_stats = team_power_index.get(away, {'attack': 1.2, 'defense': 1.0})

                # --- MODUŁ H2H (Ostatnie 3 mecze bezpośrednie) ---
                pair_key = tuple(sorted([home, away]))
                h2h_matches = h2h_database.get(pair_key, [])[-3:]  # Ostatnie 3 spotkania

                h2h_modifier_home = 1.0
                h2h_modifier_away = 1.0

                if h2h_matches:
                    h_h2h_goals = []
                    a_h2h_goals = []
                    for m in h2h_matches:
                        if m['home'] == home:
                            h_h2h_goals.append(m['home_score'])
                            a_h2h_goals.append(m['away_score'])
                        else:
                            h_h2h_goals.append(m['away_score'])
                            a_h2h_goals.append(m['home_score'])

                    avg_h_h2h = np.mean(h_h2h_goals)
                    avg_a_h2h = np.mean(a_h2h_goals)

                    # Jeśli w bezpośrednich starciach jedna drużyna wyraźnie dominowała, korygujemy współczynnik
                    if avg_h_h2h > avg_a_h2h + 0.5:
                        h2h_modifier_home = 1.1
                        h2h_modifier_away = 0.9
                    elif avg_a_h2h > avg_h_h2h + 0.5:
                        h2h_modifier_home = 0.9
                        h2h_modifier_away = 1.1

                # --- MODUŁ PRZEPAŚCI RANKINGOWEJ FIFA ---
                rank_home = team_to_rank.get(home, 25)
                rank_away = team_to_rank.get(away, 25)
                rank_diff = rank_home - rank_away

                gap_modifier_home = 1.0
                gap_modifier_away = 1.0

                if rank_diff >= 20:
                    gap_modifier_home = 0.8
                    gap_modifier_away = 1.25
                elif rank_diff >= 10:
                    gap_modifier_home = 0.9
                    gap_modifier_away = 1.15
                elif rank_diff <= -20:
                    gap_modifier_home = 1.25
                    gap_modifier_away = 0.8
                elif rank_diff <= -10:
                    gap_modifier_home = 1.15
                    gap_modifier_away = 0.9

                # Ostateczne lambdy Poissona z uwzględnieniem formy (10), H2H (3) i przepaści rankingowej
                l_home = max(0.2, (
                            h_stats['attack'] * 0.5 + a_stats['defense'] * 0.5) * h2h_modifier_home * gap_modifier_home)
                l_away = max(0.2, (
                            a_stats['attack'] * 0.5 + h_stats['defense'] * 0.5) * h2h_modifier_away * gap_modifier_away)

                h_goals = poisson.rvs(l_home)
                a_goals = poisson.rvs(l_away)

                if h_goals > a_goals:
                    sim_points[home] += 3
                elif h_goals < a_goals:
                    sim_points[away] += 3
                else:
                    sim_points[home] += 1
                    sim_points[away] += 1

        sorted_sim = sorted(sim_points.items(), key=lambda x: x[1], reverse=True)
        winner = sorted_sim[0][0]
        win_counts[winner] += 1

        for rank, (team, pts) in enumerate(sorted_sim[:5], 1):
            podium_counts[team] += 1

        for team, pts in sim_points.items():
            total_simulation_points[team] += pts

    # 5. Podsumowanie
    ranking_results = []
    for team in top_50:
        ranking_results.append({
            'team': team,
            'total_pts': total_simulation_points[team],
            'avg_pts': total_simulation_points[team] / num_simulations,
            'wins': win_counts[team],
            'top5_count': podium_counts[team]
        })

    final_df = pd.DataFrame(ranking_results).sort_values(by='total_pts', ascending=False)

    print("=" * 82)
    print("🌟 ULTIMATE POWER RANKING: TOP 5 FAWORYTÓW (FORMA 10M + H2H 3M + WYCENY + PRZEPAŚĆ)")
    print("=" * 82)
    for idx, row in enumerate(final_df.head(6).itertuples(), 1):
        win_prob = (row.wins / num_simulations) * 100
        print(
            f" {idx}. 🥇 {row.team:<18} | Suma pkt w 50 sim: {row.total_pts:6.0f} | Śr. pkt/turniej: {row.avg_pts:5.1f} | Wygrane ligi: {win_prob:4.1f}%")
    print("=" * 82 + "\n")


if __name__ == '__main__':
    run_advanced_form_h2h_power_ranking()