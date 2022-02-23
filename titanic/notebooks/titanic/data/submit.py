import kaggle
import pandas as pd
from os import environ, join


def submit(df: pd.DataFrame, project: str, message=""):
    if len(df.columns) != 2:
        raise ValueError("Dataframe for submission must have only required fields")
    file = join(environ("kaggle"), project, "data", "submission.csv")
    df.to_csv(file, index=False)

    kaggle.api.competition_submit(file, message, project)
