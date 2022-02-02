"""
see explanations of this actions in data_wrangling notebook
"""

import pandas as pd
import numpy as np
import math


def wrangling(df: pd.DataFrame, isTest: bool = False) -> pd.DataFrame:
    # drop passenger id

    df = df.drop("PassengerId", axis=1)

    # fill skipped embarked info

    if isTest:
        df.iloc[61, 6] = 1
        df.iloc[829, 6] = 1
        df.iloc[61, 10] = "S"
        df.iloc[829, 10] = "S"

    # repalce empty Fare with mean for class

    fare_class = df.groupby("Pclass").Fare.mean()
    df.Fare = df[["Pclass", "Fare"]].apply(
        lambda c: fare_class[c.Pclass] if c.Fare == 0 or pd.isna(c.Fare) else c.Fare,
        axis=1,
    )

    # create separate feature for name title

    df.Name = df.Name.str.replace("Mlle", "Miss")
    df.Name = df.Name.str.replace("Mme", "Mrs")
    df["Title"] = df.Name.apply(
        lambda n: str(n)[str(n).find(",") + 1 :].strip().split(" ")[0][:-1]
    )
    df.Title = df.Title.replace("th", "Countess")
    df.Title = df.Title.replace("Ms", "Miss")

    # fill empty age with mean of age by title

    title_age = df.groupby("Title").Age.mean().round()
    df.Age = df[["Title", "Age"]].apply(
        lambda a: title_age[a.Title] if math.isnan(a.Age) else a.Age, axis=1
    )

    #  most of cabin data are missing so we will put there the 'X' mark

    df["CabLet"] = df.Cabin.astype(str).str[0].replace("n", "X")

    # then we will assign cabin letter based on fare

    cabLet_fare_m = df[["Fare", "CabLet"]].groupby("CabLet").mean()
    cabLet_fare_q = df[["Fare", "CabLet"]].groupby("CabLet").quantile(0.75)

    def assingCabinBasedOnFare(cf: pd.DataFrame) -> str:
        cabin = cf[0]
        fare = cf[1]

        if cabin != "X":
            return cabin
        for c in cabLet_fare_q.index.values[::-1][1:-1]:
            if fare <= cabLet_fare_q.loc[c].Fare:
                return c
            else:
                return "B"

    df["CabLet"] = df[["CabLet", "Fare"]].apply(assingCabinBasedOnFare, axis=1)

    # combine SibSp and Parch into two new features

    df["Alone"] = df[["SibSp", "Parch"]].apply(
        lambda p: 0 if (p[0] + p[1] != 0) else 1, axis=1
    )
    df["Familiars"] = df.SibSp + df.Parch

    # first letter from ticket feature

    df["TicketLetter"] = df.Ticket.apply(
        lambda t: t[0] if not str(t).isnumeric() or pd.isna(t) else "X"
    )

    # add length of name feature

    df["LenName"] = df.Name.apply(len)

    # Drop redundant columns

    if isTest:
        df = df.drop("Survived", axis=1)
    df = df.drop("Name", axis=1)
    df = df.drop("Cabin", axis=1)
    df = df.drop("Ticket", axis=1)
    return df
