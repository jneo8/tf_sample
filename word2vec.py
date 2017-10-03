"""Word2vec with tensorflow."""

import os
import pandas as pd

from utils import settings

# logger
from logger import logconf
logger = logconf.Logger(__name__)


def read_data():
    """Read origin data use pandas."""
    source_list = [
        '獸性老公吻上癮.txt'
    ]

    source_list = [os]

