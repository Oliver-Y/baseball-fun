import pandas as pd
import pybaseball as ps
from datetime import date, timedelta
import random
from typing import Tuple

ps.cache.enable()

def pull_single_random_pitch_data(year=2024) -> Tuple[str, pd.DataFrame]: 
    #Given a date generate a random statcast range
    format_str = "%Y-%m-%d"
    range_days = (date(year, 10, 1) - date(year, 4, 16)).days
    rand_offset = random.randint(0, max(0, range_days))
    random_date = date(year, 4, 1) + timedelta(days=rand_offset)
    statcast_date_str = random_date.strftime(format_str)

    #Pick random game between April and October
    statcast = ps.statcast(statcast_date_str) 
    return (statcast_date_str, statcast.sample(n=1))

#Pull down down season average for pitcher
def pull_pitch_data_for_pitcher(player_id):
    pass

#Pull down pitch data for a game
# Dictionary of pitcher: [pitch_data] 
def pull_pitch_data_for_game(game_id):
    pass 

if __name__ == "__main__":
    # Test some common player searches

    date, stat = pull_single_random_pitch_data()
    print(f"Date: {date}")
    exit()
    test_searches = [
        ("skubal", "tarik")
    ]
    
    for last, first in test_searches:
        print(f"\n{'='*60}")
        print(f"SEARCHING FOR: {first.title()} {last.title()}")
        print(f"{'='*60}")
        
        result = ps.playerid_lookup(last, first, fuzzy=False)
        
        if len(result) > 0:
            print(f"\nFound {len(result)} match(es):")
            print("-" * 40)
            
            for idx, row in result.iterrows():
                print(f"Player: {row['name_first']} {row['name_last']}")
                print(f"MLB ID: {row['key_mlbam']}")
                print(f"Position: {row.get('pos', 'N/A')}")
                print(f"Team: {row.get('mlb_played_first', 'N/A')}")
                print("-" * 40)
            
            # Get pitcher stats for the first match
            if len(result) > 0:
                player_id = result["key_mlbam"].iloc[0]
                print(f"\nFetching pitcher stats for {result['name_first'].iloc[0]} {result['name_last'].iloc[0]} (ID: {player_id})...")
                
                try:
                    pitcher = ps.statcast_pitcher("2024-03-31", "2024-08-01", player_id)
                    if not pitcher.empty:
                        print(f"Columns list: {pitcher.columns.to_list()}")
                        print(f"Info: {pitcher.info()}")
                        print(f"Describe: {pitcher.describe()}")
                        
                        # Display a full sample row
                        print(f"\n{'='*80}")
                        print("FULL SAMPLE ROW (First Row):")
                        print(f"{'='*80}")
                        sample_row = pitcher.iloc[0]
                        for col, value in sample_row.items():
                            print(f"{col:50}: {value}")
                        print(f"{'='*80}")
                    
                    else:
                        print("No pitcher data found for this date range")
                except Exception as e:
                    print(f"Error fetching pitcher data: {e}")
        else:
            print("No matches found")
        
        print(f"\n{'='*60}")