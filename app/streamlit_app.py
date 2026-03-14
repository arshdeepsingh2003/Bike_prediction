import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import plotly.express as px
import os

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="Bike Demand Prediction Dashboard",
    page_icon="🚴",
    layout="wide"
)

# ------------------------------------------------
# LOAD MODEL (CACHED FOR FAST DEPLOYMENT)
# ------------------------------------------------
@st.cache_resource
def load_model():

    model_path = os.path.join("models", "xgboost_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_model()

# ------------------------------------------------
# TITLE
# ------------------------------------------------
st.title("🚴 Bike Rental Demand Prediction")

# ------------------------------------------------
# INPUT PANEL
# ------------------------------------------------
st.subheader("⚙️ Input Parameters")

col1, col2, col3, col4 = st.columns(4)

with col1:
    date = st.date_input("Date", datetime.date.today())

with col2:
    hour = st.slider("Hour", 0, 23, 12)

with col3:
    season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

with col4:
    holiday = st.selectbox("Holiday", ["No Holiday", "Holiday"])


col5, col6, col7, col8 = st.columns(4)

with col5:
    temperature = st.number_input("Temperature (°C)", -30.0, 50.0, 15.0)

with col6:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 50.0)

with col7:
    wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0)

with col8:
    visibility = st.number_input("Visibility (10m)", 0, 2000, 1000)


col9, col10, col11, col12 = st.columns(4)

with col9:
    dew_point = st.number_input("Dew Point Temperature (°C)", -30.0, 40.0, 10.0)

with col10:
    solar_radiation = st.number_input("Solar Radiation (MJ/m2)", 0.0, 5.0, 0.5)

with col11:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 50.0, 0.0)

with col12:
    snowfall = st.number_input("Snowfall (cm)", 0.0, 50.0, 0.0)


functioning_day = st.selectbox("Functioning Day", ["Yes", "No"])

st.divider()

# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------
day = date.day
month = date.month
year = date.year
weekday = date.strftime("%A")

holiday_map = {"No Holiday": 0, "Holiday": 1}
functioning_map = {"Yes": 1, "No": 0}

holiday = holiday_map[holiday]
functioning_day = functioning_map[functioning_day]

# Base features
base_data = [
    hour,
    temperature,
    humidity,
    wind_speed,
    visibility,
    dew_point,
    solar_radiation,
    rainfall,
    snowfall,
    holiday,
    functioning_day,
    day,
    month,
    year
]

columns = [
"Hour",
"Temperature(°C)",
"Humidity(%)",
"Wind speed (m/s)",
"Visibility (10m)",
"Dew point temperature(°C)",
"Solar Radiation (MJ/m2)",
"Rainfall(mm)",
"Snowfall (cm)",
"Holiday",
"Functioning Day",
"Day",
"Month",
"Year"
]

df_input = pd.DataFrame([base_data], columns=columns)

# ------------------------------------------------
# SEASON ENCODING
# ------------------------------------------------
season_cols = ['Spring', 'Summer', 'Winter']

season_data = np.zeros((1, len(season_cols)))

df_season = pd.DataFrame(season_data, columns=season_cols)

if season in season_cols:
    df_season[season] = 1

# ------------------------------------------------
# WEEKDAY ENCODING
# ------------------------------------------------
weekday_cols = ['Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday']

weekday_data = np.zeros((1, len(weekday_cols)))

df_weekday = pd.DataFrame(weekday_data, columns=weekday_cols)

if weekday in weekday_cols:
    df_weekday[weekday] = 1

# ------------------------------------------------
# FINAL DATA
# ------------------------------------------------
final_df = pd.concat([df_input, df_season, df_weekday], axis=1)

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------
st.subheader("🔮 Prediction")

predict_button = st.button("Predict Bike Demand 🚲")

if predict_button:

    scaled_data = scaler.transform(final_df)

    prediction = model.predict(scaled_data)

    demand = int(prediction[0])

    colA, colB, colC = st.columns(3)

    colA.metric("Predicted Bike Demand", f"{demand} Bikes")

    if demand < 300:
        colB.metric("Demand Level", "Low 📉")
    elif demand < 800:
        colB.metric("Demand Level", "Moderate 📊")
    else:
        colB.metric("Demand Level", "High 🔥")

    colC.metric("Selected Hour", hour)

    st.divider()

    # ------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------
    hours = list(range(24))

    simulated = [max(0, demand + np.random.randint(-150, 150)) for _ in hours]

    df_chart = pd.DataFrame({
        "Hour": hours,
        "Predicted Demand": simulated
    })

    fig = px.line(
        df_chart,
        x="Hour",
        y="Predicted Demand",
        markers=True,
        title="Estimated Demand Trend Across the Day"
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# SHOW INPUT DATA
# ------------------------------------------------
with st.expander("📄 View Processed Input Data"):
    st.dataframe(final_df)