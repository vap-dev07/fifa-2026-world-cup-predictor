import pandas as pd
import os

# Define path to the predictions file
PREDICTIONS_PATH = os.path.join("data", "processed", "predictions_output.csv")

def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"❌ Error: Could not find predictions file at {PREDICTIONS_PATH}")
        print("Make sure you've run 'python src/train_model.py' first!")
        return None
    return pd.read_csv(PREDICTIONS_PATH)

def display_top_contenders(df, top_n=15):
    print("\n" + "="*60)
    print(f"🏆 TOP {top_n} HIGHEST WIN PROBABILITIES IN THE TEST SET")
    print("="*60)
    
    # Because there are duplicate country rows for different simulated match contexts,
    # we take the highest probability instance for each unique team to see their peak potential.
    top_teams = df.sort_values(by="winner_probability", ascending=False).drop_duplicates(subset=["team_name"]).head(top_n)
    
    print(f"{'Rank':<5}{'Team':<18}{'Code':<8}{'Confederation':<15}{'Win Prob':<10}")
    print("-" * 60)
    for i, (_, row) in enumerate(top_teams.iterrows(), 1):
        print(f"{i:<5}{row['team_name']:<18}{row['country_code']:<8}{row['confederation']:<15}{row['winner_probability']:.2%}")

def search_team(df):
    print("\n" + "="*60)
    print("🔍 INDIVIDUAL TEAM LOOKUP")
    print("="*60)
    team_input = input("Enter a country name to search (e.g., USA, Mexico, Canada): ").strip()
    
    # Case-insensitive search match
    matches = df[df['team_name'].str.contains(team_input, case=False, na=False)]
    
    if matches.empty:
        print(f"❌ No matches found for '{team_input}'. Check your spelling or try another country!")
    else:
        print(f"\nFound {len(matches)} simulation contexts for '{team_input}':")
        print(f"{'Team':<18}{'Code':<8}{'Win Prob':<12}{'Predicted Winner?':<15}")
        print("-" * 60)
        for _, row in matches.sort_values(by="winner_probability", ascending=False).iterrows():
            is_winner = "Yes ✅" if row['predicted_winner'] == 1 else "No ❌"
            print(f"{row['team_name']:<18}{row['country_code']:<8}{row['winner_probability']:.2%}{is_winner:<15}")

if __name__ == "__main__":
    df = load_predictions()
    if df is not None:
        # 1. Run the top 15 report
        display_top_contenders(df, top_n=15)
        # 2. Run the interactive search
        search_team(df)