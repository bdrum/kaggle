import pandas as pd
from typing import Tuple


def from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    It assumed that data path is '../data/train.csv' and '../data/test.csv'
    """
    return (pd.read_csv(r"../data/train.csv"), pd.read_csv(r"../data/test.csv"))
