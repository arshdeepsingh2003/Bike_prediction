from src.data_preprocessing import load_data, feature_engineering
from src.feature_engineering import encode_features
from src.train_model import train_model


data_path = "data/SeoulBikeData.csv"

df = load_data(data_path)

df = feature_engineering(df)

df = encode_features(df)

train_model(df)