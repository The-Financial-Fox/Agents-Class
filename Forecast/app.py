import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os
from groq import Groq
from dotenv import load_dotenv

# 🌱 Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📈 Forecast Revenue with Prophet", page_icon="📊", layout="wide")
st.title("📈 Revenue Forecasting with Prophet")
st.markdown("Upload your Excel file with `Date` and `Revenue` columns to forecast future revenue using Prophet.")

# 📤 File Upload
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # ✅ Basic Validation
    if not {'Date', 'Revenue'}.issubset(df.columns):
        st.error("The file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # 🧹 Data Preprocessing
    df = df[['Date', 'Revenue']].dropna()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    st.subheader("📊 Uploaded Data")
    st.dataframe(df)

    # ⏱ Forecasting with Prophet
    model = Prophet()
    model.fit(df)

    future_periods = st.slider("Select forecast period (months)", 1, 24, 6)
    future = model.make_future_dataframe(periods=future_periods * 30, freq='D')
    forecast = model.predict(future)

    # 📈 Plot Forecast
    st.subheader("📈 Forecast Plot")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("🔍 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # 🧠 AI Analysis using Groq
    st.subheader("🤖 AI Analysis of Forecast")
    client = Groq(api_key=GROQ_API_KEY)

    json_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_periods * 30).to_json(orient="records")

    prompt = f"""
    You are an expert financial forecaster. Given the following Prophet model forecast data in JSON:
    {json_forecast}

    Please provide:
    - Key trends observed in the forecast.
    - Risks or uncertainties.
    - Summary insights a CFO would care about.
    - Strategic recommendations based on the forecast.
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a strategic financial advisor and forecaster."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    ai_insight = response.choices[0].message.content
    st.markdown("### 🧾 AI Forecast Commentary")
    st.write(ai_insight)
