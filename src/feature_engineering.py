import pandas as pd

def encode_features(df):

    df["Holiday"] = df["Holiday"].map({"No Holiday":0,"Holiday":1})
    df["Functioning Day"] = df["Functioning Day"].map({"Yes":1,"No":0})

    season_df = pd.get_dummies(df["Seasons"], drop_first=True)
    weekday_df = pd.get_dummies(df["weekday"], drop_first=True)

    df = pd.concat([df, season_df, weekday_df], axis=1)

    df.drop(["Seasons","weekday"], axis=1, inplace=True)

    return df