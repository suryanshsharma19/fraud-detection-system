import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
import time

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"

st.title("Real-Time Fraud Detection System")
st.markdown("---")

with st.sidebar:
    st.header("Controls")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            st.success("API Online")
            api_status = "online"
        else:
            st.error("API Error")
            api_status = "error"
    except:
        st.error("API Offline")
        api_status = "offline"
    
    if st.button("Refresh Dashboard"):
        st.experimental_rerun()

    st.markdown("---")

if api_status == "online":
    try:
        health_response = requests.get(f"{API_BASE_URL}/health")
        health_data = health_response.json()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System Status",
                health_data.get('status', 'Unknown').title(),
                delta=None
            )
        
        with col2:
            st.metric(
                "Total Predictions",
                f"{health_data.get('total_predictions', 0):,}",
                delta=None
            )
        
        with col3:
            fraud_rate = (health_data.get('fraud_detected_today', 0) /
                         max(health_data.get('total_predictions', 1), 1) * 100)
            st.metric(
                "Daily Fraud Rate",
                f"{fraud_rate:.2f}%",
                delta=None
            )
        
        with col4:
            resp_time = health_data.get('avg_response_time_ms', 0)
            st.metric(
                "Avg Response Time",
                f"{resp_time:.1f}ms",
                delta=None
            )
    
    except Exception as e:
        st.error(f"API seems down: {e}")  # happens sometimes

st.header("Single Transaction Analysis")

with st.expander("Test a Transaction", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        amt = st.number_input("Amount ($)", min_value=0.01, value=1500.00)
        merchant_cat = st.selectbox("Merchant Category", ["5411", "5812", "5732", "5541", "5999"])  # common codes
        transaction_type = st.selectbox("Transaction Type", ["purchase", "withdrawal", "transfer", "payment"])
        location = st.text_input("Location", value="New York")
        user_age = st.slider("User Age", 18, 80, 35)
    
    with col2:
        time_of_day = st.slider("Time of Day (Hour)", 0, 23, 14)
        day_of_week = st.slider("Day of Week (0=Monday)", 0, 6, 2)
        account_balance = st.number_input("Account Balance ($)", min_value=0.0, value=5000.00)
        num_transactions_day = st.number_input("Transactions Today", min_value=0, value=3)
        avg_transaction_amount = st.number_input("Average Transaction Amount ($)", min_value=0.01, value=250.00)
    
    if st.button("Analyze Transaction", type="primary"):
        transaction_data = {
            "amount": amt,
            "merchant_category": merchant_cat,
            "transaction_type": transaction_type,
            "location": location,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "user_age": user_age,
            "account_balance": account_balance,
            "num_transactions_day": num_transactions_day,
            "avg_transaction_amount": avg_transaction_amount
        }
        
        if api_status == "online":
            try:
                with st.spinner("Analyzing transaction..."):
                    response = requests.post(
                        f"{API_BASE_URL}/predict",
                        json=transaction_data,
                        timeout=10
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fraud_status = "FRAUD DETECTED" if result['is_fraud'] else "LEGITIMATE"
                        color = "red" if result['is_fraud'] else "green"
                        st.markdown(f"<h3 style='color: {color};'>{fraud_status}</h3>", unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")
                    
                    with col3:
                        st.metric("Risk Score", f"{result['risk_score']}/100")
                    
                    col4, col5 = st.columns(2)
                    
                    with col4:
                        st.metric("Confidence", f"{result['confidence']:.4f}")
                    
                    with col5:
                        st.metric("Processing Time", f"{result['processing_time_ms']:.2f}ms")
                    
                    if 'individual_predictions' in result:
                        st.subheader("Model Breakdown")
                        pred_df = pd.DataFrame(result['individual_predictions']).T
                        st.dataframe(pred_df)
                
                else:
                    st.error(f"Prediction failed: {response.status_code}")
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.error("API is offline. Cannot make predictions.")

st.header("Batch Analysis")

uploaded_file = st.file_uploader("Upload CSV file with transactions", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        if st.button("Analyze Batch", type="primary"):
            if api_status == "online":
                with st.spinner("Processing batch..."):
                    transactions = []
                    for _, row in df.iterrows():
                        transactions.append({"data": row.to_dict(), "type": "general"})
                    
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/predict/batch",
                            json={"transactions": transactions},
                            timeout=30
                        )
                        
                        if response.status_code == 200:
                            results = response.json()
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total Processed", results['total_processed'])
                            
                            with col2:
                                st.metric("Fraud Detected", results['fraud_detected'])
                            
                            with col3:
                                st.metric("Processing Time", f"{results['processing_time_ms']:.2f}ms")
                            
                            predictions_df = pd.DataFrame(results['predictions'])
                            st.subheader("Results")
                            st.dataframe(predictions_df)
                            
                            fraud_chart = px.pie(
                                values=[results['fraud_detected'], results['total_processed'] - results['fraud_detected']],
                                names=['Fraud', 'Legitimate'],
                                title="Fraud Distribution"
                            )
                            st.plotly_chart(fraud_chart)
                        
                        else:
                            st.error(f"Batch prediction failed: {response.status_code}")
                    
                    except Exception as e:
                        st.error(f"Error processing batch: {e}")
            else:
                st.error("API is offline. Cannot process batch.")
    
    except Exception as e:
        st.error(f"Error reading file: {e}")

st.header("System Analytics")

if api_status == "online":
    try:
        analytics_response = requests.get(f"{API_BASE_URL}/analytics/summary")
        if analytics_response.status_code == 200:
            analytics = analytics_response.json()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Daily Statistics")
                daily_data = analytics.get('daily_stats', {})
                for key, value in daily_data.items():
                    st.metric(key.replace('_', ' ').title(), value)
            
            with col2:
                st.subheader("Model Performance")
                model_data = analytics.get('model_performance', {})
                for key, value in model_data.items():
                    st.metric(key.replace('_', ' ').title(), f"{value:.4f}" if isinstance(value, float) else value)
        
        else:
            st.warning("Analytics data not available")
    
    except Exception as e:
        st.error(f"Error fetching analytics: {e}")
else:
    st.warning("Connect to API to view analytics")

st.markdown("---")
st.caption("Fraud Detection System - Real-time monitoring and analysis")
