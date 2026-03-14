# 🚴 Bike Demand Prediction

A Machine Learning project that predicts **hourly bike rental demand** using weather conditions, time information, and seasonal factors.
The model is trained on the **Seoul Bike Rental Dataset** and deployed as an **interactive Streamlit web application**.

---

# 📌 Project Overview

Bike rental services must anticipate demand to ensure bikes are available when needed.
This project builds a **machine learning model** to predict the number of bikes rented at a given hour based on several features like:

* Weather conditions
* Time and date
* Holiday status
* Seasonal factors

The model helps understand **when bike demand will be high or low**, which can help optimize bike availability and operations.

---

# 🧠 Machine Learning Workflow

The project follows a standard **ML pipeline**:

1. **Business Problem Definition**
2. **Data Collection**
3. **Exploratory Data Analysis (EDA)**
4. **Feature Engineering**
5. **Handling Multicollinearity**
6. **Data Preprocessing & Scaling**
7. **Model Training**
8. **Hyperparameter Tuning**
9. **Model Evaluation**
10. **Model Deployment**

---

# 📊 Dataset

Dataset used:

**Seoul Bike Rental Dataset**

It contains hourly information including:

* Date
* Hour
* Temperature
* Humidity
* Wind Speed
* Visibility
* Dew Point Temperature
* Solar Radiation
* Rainfall
* Snowfall
* Seasons
* Holiday
* Functioning Day
* Rented Bike Count (Target)

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Streamlit
* Plotly

---

# 🏗 Project Structure

```
Bike_prediction
│
├── app
│   └── streamlit_app.py
│
├── models
│   ├── xgboost_model.pkl
│   └── scaler.pkl
│
├── data
│   └── SeoulBikeData.csv
│
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│
├── requirements.txt
└── README.md
```

---

# 📈 Model Performance

The final model selected was **XGBoost Regressor**.

### Training Performance

| Metric   | Value |
| -------- | ----- |
| R² Score | 0.991 |
| RMSE     | 61.63 |
| MAE      | 41.27 |

### Testing Performance

| Metric   | Value  |
| -------- | ------ |
| R² Score | 0.936  |
| RMSE     | 162.32 |
| MAE      | 98.59  |

The model shows **excellent generalization performance**.

---

# 🚀 Running the Project Locally

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/bike-demand-prediction.git
```

### 2️⃣ Navigate to the project

```
cd bike-demand-prediction
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit application

```
streamlit run app/streamlit_app.py
```

---

# 🌐 Web Application

The Streamlit app allows users to:

* Enter weather conditions
* Select date and time
* Predict bike demand instantly

The dashboard provides:

* Predicted bike demand
* Demand level indicator
* Visualization of demand trend

---

# 📷 Application Interface

The dashboard includes:

* Professional input panels
* Demand prediction cards
* Interactive demand charts

---

# 🔮 Future Improvements

Possible improvements for this project include:

* Integrating a **weather API** for real-time weather data
* Adding **feature importance visualization**
* Creating **demand heatmaps**
* Deploying the model as a **REST API**
* Using **real-time bike station data**

---

# 👨‍💻 Author

Developed as a **Machine Learning project for bike demand prediction and deployment using Streamlit**.

---

# ⭐ If you like this project

Give it a ⭐ on GitHub!
