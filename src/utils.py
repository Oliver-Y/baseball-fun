import pandas as pd
import pybaseball as ps
from datetime import date, timedelta
import random
import logging 
from typing import Tuple, Optional

ps.cache.enable()

# Configure root logger once at startup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Get a module-level logger
logger = logging.getLogger(__name__)

def get_statcast_season_range(year: int) -> Tuple[date, date]:
    """
    Return a reasonable (April 1 – Oct 1) Statcast date range for a given season.
    Ensures we don’t hit empty pre/postseason windows.
    """
    start = date(year, 4, 1)   # Opening month
    end   = date(year, 10, 1)  # Before playoffs
    return start, end


def random_statcast_date(year: int) -> date:
    """
    Pick a random in-season date that likely has data.
    """
    start, end = get_statcast_season_range(year)
    delta_days = (end - start).days
    offset = random.randint(0, delta_days)
    return start + timedelta(days=offset)

def pull_single_random_pitch_data(year=2024) -> Tuple[str, pd.DataFrame]: 
    format_str = "%Y-%m-%d"
    random_date = random_statcast_date(year)
    statcast_date_str = random_date.strftime(format_str)
    #Pick random game between April and October
    statcast = ps.statcast(statcast_date_str) 
    return (statcast_date_str, statcast.sample(n=1))

#

def pull_pitch_data_for_pitcher(last, first, year = 2024) -> Optional[pd.DataFrame]:
    format_str = "%Y-%m-%d"
    result = ps.playerid_lookup(last, first, fuzzy=False)
    logger.info(f"\nFetching pitcher stats for {result['name_first'].iloc[0]} {result['name_last'].iloc[0]}...")
    if len(result) <= 0: 
        return None
    logger.info(f"Result: {result}")
    player_id = result["key_mlbam"].iloc[0]
    start_date, end_date = get_statcast_season_range(2023)
    logger.info(f"Start Date: {start_date.strftime(format_str)}, End Date: {end_date.strftime(format_str)}")
    pitcher = ps.statcast_pitcher(start_date.strftime(format_str), end_date.strftime(format_str), player_id)
    if not pitcher.empty:
        return pitcher
    return None

#Pull down pitch data for a game
def pull_pitch_data_for_game(game_id):
    pass 

if __name__ == "__main__":
    single_date, stat = pull_single_random_pitch_data()
    logger.info(f"Date: {single_date}, Stat: {stat}")
    pitcher = pull_pitch_data_for_pitcher("skubal", "tarik")
    logger.info(f"Pitcher: {pitcher}")
    exit()