import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import titanic.data.load
import titanic.data.wrangling as wrng
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


train_df_orig, test_df_orig = titanic.data.load.from_csv()

train_df = wrng.wrangling(train_df_orig)
test_df = wrng.wrangling(test_df_orig)

X_train, X_test, y_train, y_test = train_test_split(
    train_df, train_df_orig.Survived, test_size=0.2, random_state=50
)

numeric_features = ["Pclass", "Alone", "Familiars", "LenName"]
numeric_outliers = ["Age", "Fare"]
numeric_transformer = MinMaxScaler()
numeric_outliers_transformer = RobustScaler()

categorical_features = ["Embarked", "Sex", "Title", "CabLet", "TicketLetter"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("out", numeric_outliers_transformer, numeric_outliers),
        ("cat", categorical_transformer, categorical_features),
    ]
)

n_est = [800, 900, 1000]
max_depth = [2, 5, 10]
max_depth.append(None)


param_grid = {
    # Number of trees in random forest
    "rand_for__n_estimators": n_est,
    # Number of features to consider at every split
    "rand_for__max_features": ["auto", "sqrt"],
    # Maximum number of levels in tree
    "rand_for__max_depth": max_depth,
    # Minimum number of samples required to split a node
    "rand_for__min_samples_split": [2, 5, 10],
    # Minimum number of samples required at each leaf node
    "rand_for__min_samples_leaf": [2, 4, 8],
    # Method of selecting samples for training each tree
    "rand_for__bootstrap": [True, False],
}


clf = Pipeline(
    steps=[("preprocessor", preprocessor), ("rand_for", RandomForestClassifier())]
)

grid_search = GridSearchCV(clf, param_grid, cv=10, n_jobs=10)  # , verbose=5)

grid_search.fit(X_train, y_train)

print("model score: %.3f" % grid_search.score(X_test, y_test))
