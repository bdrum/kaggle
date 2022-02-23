"""
see explanations of this actions in data_wrangling notebook
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


def preprocessing(
    num_feat: list = ["Pclass", "Alone", "Familiars", "LenName"],
    num_outl: list = ["Age", "Fare"],
    categ_feat: list = ["Embarked", "Sex", "Title", "CabLet", "TicketLetter"],
) -> ColumnTransformer:
    numeric_features = num_feat
    numeric_outliers = num_outl
    numeric_transformer = MinMaxScaler()
    numeric_outliers_transformer = Pipeline(
        steps=[("remove_out", RobustScaler()), ("num", numeric_transformer)]
    )

    categorical_features = categ_feat
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("out", numeric_outliers_transformer, numeric_outliers),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor
