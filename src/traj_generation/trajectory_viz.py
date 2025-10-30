## Pull random data and visualize
from src import utils
import logging 
from datetime import date, timedelta
from typing import Tuple, Optional

#TODO have an abstract data class that parses data into the relevant and right format

# Get a module-level logger
logger = logging.getLogger(__name__)

test_date, stat = utils.pull_single_random_pitch_data()
logger.info(f"Date: {test_date}, Stat: {stat}")
