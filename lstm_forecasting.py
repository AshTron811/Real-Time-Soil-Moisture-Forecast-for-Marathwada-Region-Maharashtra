# lstm_forecasting.py

import os
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from datetime import timedelta

WINDOW_SIZE    = 7
FORECAST_DAYS  = 30
MODEL_FILE     = "lstm_model.h5"
LAST_TRAINED   = "last_trained_date.txt"
SERIES_CSV     = "sm_series.csv"

st.title("4️⃣ LSTM Forecasting of Soil Moisture")

def load_series():
    if not os.path.exists(SERIES_CSV):
        st.error(f"{SERIES_CSV} not found — run rf_downscaling.py first."); st.stop()
    df = pd.read_csv(SERIES_CSV, parse_dates=["ds"])
    if df.empty:
        st.error(f"{SERIES_CSV} is empty."); st.stop()
    return df

def train_model(df):
    if os.path.exists(MODEL_FILE):
        m = load_model(MODEL_FILE, custom_objects={"mse":MeanSquaredError()})
        m.compile(optimizer=Adam(), loss="mse")
    else:
        m = Sequential([
            LSTM(50, activation="relu", return_sequences=True, input_shape=(WINDOW_SIZE,1)),
            LSTM(25, activation="relu"),
            Dropout(0.2),
            Dense(1)
        ])
        m.compile(optimizer="adam", loss="mse")

    last = pd.to_datetime("1900-01-01")
    if os.path.exists(LAST_TRAINED):
        try: last = pd.to_datetime(open(LAST_TRAINED).read().strip())
        except: pass

    new_data = df[df.ds > last]
    if not new_data.empty and len(df) > WINDOW_SIZE:
        start_idx = max(0, new_data.index[0] - WINDOW_SIZE)
        tr = df.iloc[start_idx:]
        X, y = [], []
        for j in range(len(tr) - WINDOW_SIZE):
            X.append(tr.y.values[j:j+WINDOW_SIZE])
            y.append(tr.y.values[j+WINDOW_SIZE])
        X = np.array(X).reshape(-1, WINDOW_SIZE, 1)
        y = np.array(y)
        with st.spinner("Training LSTM…"):
            m.fit(X, y, epochs=100, batch_size=16, verbose=0)
        open(LAST_TRAINED, "w").write(df.ds.iloc[-1].strftime("%Y-%m-%d"))

    m.save(MODEL_FILE)
    return m

def validate_model(m, df):
    preds, acts = [], []
    for i in range(WINDOW_SIZE, len(df)):
        inp = df.y.values[i-WINDOW_SIZE:i].reshape(1, WINDOW_SIZE, 1)
        p   = m.predict(inp, verbose=0)[0,0]
        preds.append(p); acts.append(df.y.iloc[i])
    mse = np.mean((np.array(acts) - np.array(preds))**2)
    return {"RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(np.array(acts) - np.array(preds)))}

def generate_forecast(m, df):
    buf = list(df.y.values[-WINDOW_SIZE:])
    last_date = df.ds.iloc[-1]
    fut = []
    pbar = st.progress(0)
    for k in range(FORECAST_DAYS):
        inp = np.array(buf[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
        p   = m.predict(inp, verbose=0)[0,0]
        fut.append({"ds": last_date + timedelta(days=k+1), "y_pred": p})
        buf.append(p)
        pbar.progress((k+1)/FORECAST_DAYS)
    return pd.DataFrame(fut)

df_series = load_series()
st.subheader("Historical series"); st.line_chart(df_series.set_index("ds")["y"])

model   = train_model(df_series)
metrics = validate_model(model, df_series)
st.subheader("Validation metrics"); st.write(metrics)

df_fc = generate_forecast(model, df_series)
st.subheader(f"{FORECAST_DAYS}-Day Forecast"); st.line_chart(df_fc.set_index("ds")["y_pred"]); st.dataframe(df_fc)
