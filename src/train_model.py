import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


def evaluate(y_true, y_pred, name):

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Performance")
    print("MSE:", round(mse,3))
    print("RMSE:", round(rmse,3))
    print("MAE:", round(mae,3))
    print("R2:", round(r2,3))


def train_model(df):

    X = df.drop("Rented Bike Count", axis=1)
    y = df["Rented Bike Count"]

    x_train,x_test,y_train,y_test = train_test_split(
        X,y,test_size=0.3,random_state=42
    )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    model = XGBRegressor()

    model.fit(x_train,y_train)

    # predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # evaluate
    evaluate(y_train,y_train_pred,"Training Data")
    evaluate(y_test,y_test_pred,"Testing Data")

    # save model
    pickle.dump(model,open("models/xgboost_model.pkl","wb"))
    pickle.dump(scaler,open("models/scaler.pkl","wb"))

    print("\nModel saved successfully")