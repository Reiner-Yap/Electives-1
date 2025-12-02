import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime
from calendar import monthrange
import matplotlib.pyplot as plt

st.title("🇵🇭 Philippine City AQI Predictor (SVM)")
st.write("Predict the Air Quality Index (AQI) for a selected city and month/year using historical data from the previous year.")

st.subheader("1. Load Dataset")
try:
    dataset_path = kagglehub.dataset_download("bwandowando/philippine-major-cities-air-quality-data")
    st.success("Dataset downloaded successfully.")
except Exception as e:
    st.error(f"Failed to download dataset: {e}")
    st.stop()

# Find CSV
csv_file = None
for root, dirs, files in os.walk(dataset_path):
    for f in files:
        if f.endswith(".csv"):
            csv_file = os.path.join(root, f)
            break
if csv_file is None:
    st.error("No CSV dataset found.")
    st.stop()

df = pd.read_csv(csv_file)

df.rename(columns={
    "datetime": "Date",
    "main.aqi": "AQI",
    "components.co": "CO",
    "components.no": "NO",
    "components.no2": "NO2",
    "components.o3": "O3",
    "components.so2": "SO2",
    "components.pm2_5": "PM2_5",
    "components.pm10": "PM10",
    "components.nh3": "NH3",
    "coord.lon": "Lon",
    "coord.lat": "Lat",
    "extraction_date_time": "ExtractedAt",
    "city_name": "City"
}, inplace=True)

df["Date"] = pd.to_datetime(df["Date"])
df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")
df.dropna(subset=["AQI"], inplace=True)

st.write("### Dataset Preview")
st.dataframe(df.head())

st.subheader("2. Select City")
cities = df["City"].unique()
selected_city = st.selectbox("Choose City", cities)

st.subheader("3. Select Month and Year to Predict")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Allow 2025 for prediction
years = list(range(df["Year"].min()+1, 2026))
selected_year = st.selectbox("Select Year to Predict", sorted(years))
months = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
          7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
selected_month = st.selectbox("Select Month to Predict", list(months.values()))
month_number = [k for k,v in months.items() if v==selected_month][0]

prev_year = selected_year - 1
train_df = df[(df["City"]==selected_city) & (df["Year"]==prev_year) & (df["Month"]==month_number)]

# If previous year month data is missing, use all available city/year data
if train_df.empty:
    st.warning(f"No full month data available for {selected_month} {prev_year}. Using all available data for {selected_city} in {prev_year}.")
    train_df = df[(df["City"]==selected_city) & (df["Year"]==prev_year)]

if train_df.empty:
    st.error(f"No historical data available for {selected_city} in {prev_year}. Cannot make prediction.")
    st.stop()

st.write(f"Rows of historical data used for prediction: {len(train_df)}")
if len(train_df) < 5:
    st.warning("⚠️ Very few data points in historical month. Predictions may be inaccurate.")

train_df = train_df.sort_values("Date")
all_days = pd.DataFrame({"Date": pd.date_range(start=train_df["Date"].min(),
                                               end=train_df["Date"].max())})
train_df = pd.merge(all_days, train_df, on="Date", how="left")
train_df["AQI"].interpolate(method="linear", inplace=True)

st.write(f"### Historical Data for {selected_city} - {selected_month} {prev_year}")
st.dataframe(train_df)

train_df["DayIndex"] = train_df["Date"].dt.day  # Day of month as feature
X = train_df[["DayIndex"]].values
y = train_df["AQI"].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

st.subheader("4. Train SVM Model")
model = SVR(kernel="rbf", C=100, epsilon=0.01)
model.fit(X_scaled, y_scaled)
st.success("SVM model trained successfully!")

st.subheader(f"5. Predict AQI for {selected_month} {selected_year}")
days_in_month = monthrange(selected_year, month_number)[1]
future_dates = pd.date_range(start=datetime(selected_year, month_number, 1), periods=days_in_month, freq='D')
future_day_index = np.arange(1, days_in_month + 1).reshape(-1,1)
X_future_scaled = scaler_X.transform(future_day_index)
pred_scaled = model.predict(X_future_scaled)
pred_aqi = scaler_y.inverse_transform(pred_scaled.reshape(-1,1)).flatten()

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_AQI": pred_aqi
})

st.write(f"### Predicted AQI for {selected_city} - {selected_month} {selected_year}")
st.dataframe(future_df)

csv = future_df.to_csv(index=False)
st.download_button("📥 Download Predicted AQI CSV", csv, f"AQI_{selected_city}_{selected_month}_{selected_year}.csv")

st.subheader("6. Visualization")
fig, ax = plt.subplots(figsize=(8,4))
ax.plot(train_df["Date"], train_df["AQI"], label=f"Historical AQI ({prev_year})")
ax.plot(future_df["Date"], future_df["Predicted_AQI"], label=f"Predicted AQI ({selected_year})")
ax.set_xlabel("Date")
ax.set_ylabel("AQI")
ax.set_title(f"{selected_city} - {selected_month} AQI Prediction")
ax.legend()
st.pyplot(fig)

st.subheader("7. RMSE on Historical Data")
y_pred_train_scaled = model.predict(X_scaled)
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1,1)).flatten()
rmse = np.sqrt(mean_squared_error(y, y_pred_train))
st.metric("RMSE (Predictions on Historical Data)", f"{rmse:.2f}")
