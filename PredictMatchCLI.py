import os
import pandas as pd
from datetime import datetime


def add_match_to_predictions():
    predictions_file = 'predictions.csv'

    # Standardowe kolumny z pliku results.csv
    expected_columns = [
        'date', 'home_team', 'away_team',
        'home_score', 'away_score',
        'tournament', 'city', 'country', 'neutral'
    ]

    print("=" * 60)
    print("⚽ KREATOR PREDYKCJI MECZÓW (CLI DLA PYCHARM)")
    print("=" * 60)
    print("Wprowadź dane nadchodzącego meczu, który chcesz sprawdzić.\n")

    # Pobieranie danych od użytkownika
    while True:
        date_str = input("Podaj datę meczu (format RRRR-MM-DD, np. 2026-06-11): ").strip()
        try:
            # Walidacja formatu daty
            parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
            break
        except ValueError:
            print("❌ Błędny format daty! Użyj formatu RRRR-MM-DD.")

    home_team = input("Podaj reprezentację GOSPODARZA (np. Poland, Argentina): ").strip()
    away_team = input("Podaj reprezentację GOŚCIA (np. Germany, Brazil): ").strip()
    tournament = input("Podaj nazwę turniej / rozgrywek (np. World Cup, Friendly): ").strip()
    city = input("Podaj miasto (opcjonalnie, np. New York): ").strip()
    country = input("Podaj kraj gospodarza turnieju (opcjonalnie, np. USA): ").strip()

    neutral_input = input("Czy mecz na neutralnym terenie? (t/n, domyślnie n): ").strip().lower()
    is_neutral = True if neutral_input in ['t', 'tak', 'y', 'yes'] else False

    # Ponieważ to mecz nadchodzący, wyniki (home_score, away_score) ustawiamy domyślnie na 0 lub puste (np. NaN)
    new_match = {
        'date': date_str,
        'home_team': home_team,
        'away_team': away_team,
        'home_score': 0,  # Wartość domyślna, algorytm przewidzi wynik
        'away_score': 0,  # Wartość domyślna, algorytm przewidzi wynik
        'tournament': tournament if tournament else 'Friendly',
        'city': city if city else 'Unknown',
        'country': country if country else home_team,
        'neutral': is_neutral
    }

    # Sprawdzamy czy plik predictions.csv już istnieje
    if os.path.exists(predictions_file):
        try:
            pred_df = pd.read_csv(predictions_file)
            # Jeśli plik ma inne kolumny lub jest pusty, inicjalizujemy od nowa
            if list(pred_df.columns) != expected_columns:
                pred_df = pd.DataFrame(columns=expected_columns)
        except Exception:
            pred_df = pd.DataFrame(columns=expected_columns)
    else:
        pred_df = pd.DataFrame(columns=expected_columns)

    # Dodajemy nowy wiersz
    new_row_df = pd.DataFrame([new_match])
    pred_df = pd.concat([pred_df, new_row_df], ignore_index=True)

    # Zapisujemy z powrotem do predictions.csv
    pred_df.to_csv(predictions_file, index=False)

    print("\n" + "—" * 60)
    print(f"✅ Sukces! Mecz **{home_team} vs {away_team}** został zapisany do '{predictions_file}'.")
    print(f"📁 Aktualna liczba meczów oczekujących w pliku: {len(pred_df)}")
    print("—" * 60)

    # Pytanie czy dodać kolejny
    cont = input("Czy chcesz dodać kolejny mecz? (t/n): ").strip().lower()
    if cont in ['t', 'tak', 'y', 'yes']:
        print("\n")
        add_match_to_predictions()


if __name__ == '__main__':
    add_match_to_predictions()