import os
import pandas as pd
import numpy as np
import unicodedata


def clean_name(name):
    """Normalizuje i czyści imiona i nazwiska do bezbłędnego łączenia tabel."""
    if pd.isna(name): return ""
    nfkd_form = unicodedata.normalize('NFKD', str(name))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).strip().lower()


def get_extreme_progressive_multiplier(rank):
    """
    Ekstremalnie agresywny współczynnik FIFA:
    - Top 3: Gigantyczna promocja (1.75 - 2.00)
    - Miejsca 4-5: Bardzo silny bonus (1.50)
    - Miejsca 6-10: Neutralna baza (1.00 - 1.20)
    - Miejsca 11-20: Szybki spadek (0.25 - 0.50)
    - Powyżej 20: Twarde odcięcie (bliskie 0.01)
    """
    if pd.isna(rank): return 0.01
    r = int(rank)
    if r <= 1:
        return 2.00
    elif r <= 3:
        return 1.75
    elif r <= 5:
        return 1.50
    elif r <= 8:
        return 1.20
    elif r <= 10:
        return 1.00
    elif r <= 15:
        return 0.50
    elif r <= 20:
        return 0.25
    else:
        return max(0.01, 0.20 - (r - 20) * 0.015)


def run_extreme_fifa_model():
    goalscorers_file = 'goalscorers.csv'
    club_stats_file = 'players_data-2024_2025.csv'

    for f in [goalscorers_file, club_stats_file]:
        if not os.path.exists(f):
            print(f"❌ Brak wymaganego pliku: '{f}'!")
            return

    print("📥 Wczytywanie bazy reprezentacyjnej i klubowej...")
    goals_df = pd.read_csv(goalscorers_file, low_memory=False)
    club_df = pd.read_csv(club_stats_file, low_memory=False)

    # 1. Przetwarzanie bazy reprezentacyjnej z wagą Time-Decay (okno 2 lat wstecz od czerwca 2026)
    goals_df['date'] = pd.to_datetime(goals_df['date'])
    ref_date = pd.to_datetime('2026-06-01')
    start_date = pd.to_datetime('2024-06-01')

    filtered_goals = goals_df[(goals_df['date'] >= start_date) & (goals_df['date'] <= ref_date)].copy()

    # Wyliczenie liczby miesięcy wstecz od daty referencyjnej turnieju
    filtered_goals['months_ago'] = (ref_date - filtered_goals['date']).dt.days / 30.4375

    # Wzór wykładniczego zaniku (Time-Decay)
    filtered_goals['time_decay_weight'] = 0.5 + 1.5 * np.exp(-0.07 * filtered_goals['months_ago'])
    filtered_goals['match_id'] = filtered_goals['date'].astype(str) + "_" + filtered_goals['home_team'] + "_" + \
                                 filtered_goals['away_team']

    if 'own_goal' in filtered_goals.columns:
        filtered_goals = filtered_goals[filtered_goals['own_goal'] != True]
        filtered_goals = filtered_goals[filtered_goals['own_goal'] != 1]

    scorer_col = 'scorer' if 'scorer' in filtered_goals.columns else 'player'
    team_col = 'team' if 'team' in filtered_goals.columns else 'nationality'

    nat_summary = filtered_goals.groupby(scorer_col).agg(
        nat_goals=('date', 'count'),
        nat_weighted_goals=('time_decay_weight', 'sum'),
        team=(team_col, 'last')
    ).reset_index()

    match_counts = filtered_goals.groupby([scorer_col, team_col])['match_id'].nunique().reset_index(
        name='matches_with_goals')
    nat_summary = nat_summary.merge(match_counts[[scorer_col, 'matches_with_goals']], on=scorer_col, how='left')
    nat_summary['nat_gpm'] = nat_summary['nat_weighted_goals'] / nat_summary['matches_with_goals'].clip(lower=1)

    # 2. Przetwarzanie bazy klubowej
    club_summary = club_df.groupby('Player').agg(
        club_goals=('Gls', 'sum'),
        club_assists=('Ast', 'sum'),
        club_xg=('xG', 'sum'),
        club_min=('Min', 'sum'),
        squad=('Squad', lambda x: ', '.join(x.unique()))
    ).reset_index()

    # 3. Łączenie tabel po znormalizowanych imionach
    nat_summary['clean_name'] = nat_summary[scorer_col].apply(clean_name)
    club_summary['clean_name'] = club_summary['Player'].apply(clean_name)

    merged = pd.merge(nat_summary, club_summary, on='clean_name', how='inner')

    # 4. Słownik rankingowy FIFA dla reprezentacji
    elite_fifa_ranks = {
        'Argentina': 1, 'France': 2, 'Spain': 3, 'England': 4,
        'Brazil': 5, 'Portugal': 6, 'Belgium': 7, 'Netherlands': 8,
        'Italy': 9, 'Germany': 10, 'Uruguay': 11, 'Croatia': 12,
        'Colombia': 13, 'Morocco': 14, 'USA': 15, 'Mexico': 16,
        'Norway': 35, 'Egypt': 36, 'Poland': 28, 'Nigeria': 38, 'Austria': 24
    }

    def get_team_rank(team_name):
        rank = elite_fifa_ranks.get(team_name, None)
        if rank is None:
            for t, r in elite_fifa_ranks.items():
                if str(team_name) in t or t in str(team_name):
                    rank = r
                    break
        return rank if rank is not None else 40

    merged['fifa_rank'] = merged['team'].map(get_team_rank)
    merged['team_multiplier'] = merged['fifa_rank'].apply(get_extreme_progressive_multiplier)

    # 5. Kalkulacja Hybrydowego Indeksu Króla Strzelców
    merged['national_score'] = merged['nat_weighted_goals'] * merged['nat_gpm']
    merged['club_score'] = merged['club_goals'] + merged['club_xg']

    merged['hybrid_golden_boot_index'] = (
            (merged['national_score'] * 0.4) +
            (merged['club_score'] * 0.4) +
            (merged['team_multiplier'] * 20.0 * 0.2)
    )

    qualified_hybrid = merged[(merged['nat_goals'] >= 3) | (merged['club_goals'] >= 12)]
    top_hybrid = qualified_hybrid.sort_values(by='hybrid_golden_boot_index', ascending=False).head(15)

    # 6. Prezentacja wyników
    print("\n" + "=" * 128)
    print(
        "🏆 FINAŁOWY RANKING ZE EKSTREMALNYM WSPÓŁCZYNNIKIEM FIFA (TOP 3 MEGA-PROMOCJA -> TWARDE ODCIĘCIE POZA TOP 20)")
    print("=" * 128)
    for idx, row in enumerate(top_hybrid.itertuples(), 1):
        p_name = getattr(row, 'Player_x', getattr(row, 'Player', 'Unknown'))
        team = row.team
        rank = int(row.fifa_rank)
        w_goals = row.nat_weighted_goals
        club_g = row.club_goals
        mult = row.team_multiplier
        index = row.hybrid_golden_boot_index

        print(
            f" {idx:2d}. ⚽ {p_name:<22} | Kadra: {str(team):<15} (Rank: {rank:<2}) | Mult: {mult:4.2f} | Gole(Ważone): {w_goals:4.1f} | Klub Gole: {club_g:2.0f} | Indeks: {index:5.1f}")
    print("=" * 128 + "\n")


if __name__ == '__main__':
    run_extreme_fifa_model()