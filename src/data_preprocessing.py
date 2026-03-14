import pandas as pd

def load_data(path):
    df = pd.read_csv(path, encoding="unicode_escape")
    return df


def feature_engineering(df):

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    df["weekday"] = df["Date"].dt.day_name()
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df.drop("Date", axis=1, inplace=True)

    return df