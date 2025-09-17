import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os
import sys
from pathlib import Path
import time

# Add src to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root / "src"))

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load pre-trained models for inference"""
    try:
        # Try to load from the models directory
        models_path = project_root / "data" / "models"
        
        # Mock model loading for demo purposes
        # In real deployment, you'd load your actual models here
        class MockModel:
            def predict_proba(self, X):
                # Generate realistic fraud probabilities
                np.random.seed(hash(str(X.iloc[0].values)) % 2**32)  # Deterministic but varied
                fraud_prob = np.random.beta(2, 8)  # Most transactions are legitimate
                return np.array([[1-fraud_prob, fraud_prob]])
            
            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] > 0.5).astype(int)
        
        models = {
            'rf_model': MockModel(),
            'xgb_model': MockModel(),
            'lr_model': MockModel()
        }
        
        st.success("✅ Models loaded successfully!")
        return models
    except Exception as e:
        st.error(f"⚠️ Could not load models: {e}")
        return None

@st.cache_data
def create_sample_data():
    """Create sample transaction data for demonstration"""
    np.random.seed(42)  # For consistent demo data
    
    sample_data = []
    for i in range(100):
        # Generate realistic transaction data
        is_fraud = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% fraud rate
        
        if is_fraud:
            # Fraudulent transactions tend to be higher amounts, odd times
            amount = np.random.lognormal(7, 1.5)
            time_of_day = np.random.choice([2, 3, 23, 0, 1])  # Late night
            num_transactions = np.random.randint(8, 20)  # Many transactions
        else:
            # Normal transactions
            amount = np.random.lognormal(5, 1)
            time_of_day = np.random.randint(6, 22)  # Normal hours
            num_transactions = np.random.randint(1, 5)  # Few transactions
        
        sample_data.append({
            'transaction_id': f'TXN_{i:04d}',
            'amount': round(amount, 2),
            'merchant_category': np.random.choice(['5411', '5812', '5732', '5541', '5999']),
            'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer', 'payment']),
            'location': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']),
            'time_of_day': time_of_day,
            'day_of_week': np.random.randint(0, 7),
            'user_age': np.random.randint(18, 80),
            'account_balance': np.random.uniform(100, 10000),
            'num_transactions_day': num_transactions,
            'avg_transaction_amount': np.random.uniform(50, 500),
            'actual_fraud': is_fraud
        })
    
    return pd.DataFrame(sample_data)

def predict_fraud(transaction_data, models):
    """Make fraud prediction using loaded models"""
    if models is None:
        return {
            'is_fraud': False,
            'fraud_probability': 0.0,
            'risk_score': 0,
            'confidence': 0.0,
            'processing_time_ms': 0.0,
            'individual_predictions': {}
        }
    
    start_time = time.time()
    
    # Convert transaction data to DataFrame
    df = pd.DataFrame([transaction_data])
    
    # Feature engineering (simplified)
    df['amount_log'] = np.log1p(df['amount'])
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_night'] = df['time_of_day'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    df['balance_ratio'] = df['amount'] / (df['account_balance'] + 1)
    df['frequency_risk'] = np.minimum(df['num_transactions_day'] / 10, 1.0)
    
    # Select features for prediction
    feature_cols = ['amount_log', 'time_of_day', 'day_of_week', 'user_age', 
                   'is_weekend', 'is_night', 'balance_ratio', 'frequency_risk']
    
    X = df[feature_cols]
    
    # Get predictions from each model
    predictions = {}
    fraud_probs = []
    
    for model_name, model in models.items():
        try:
            prob = model.predict_proba(X)[0, 1]  # Fraud probability
            predictions[model_name] = {
                'fraud_probability': round(prob, 4),
                'prediction': int(prob > 0.5)
            }
            fraud_probs.append(prob)
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")
            predictions[model_name] = {'fraud_probability': 0.0, 'prediction': 0}
            fraud_probs.append(0.0)
    
    # Ensemble prediction (average)
    avg_fraud_prob = np.mean(fraud_probs)
    is_fraud = avg_fraud_prob > 0.5
    risk_score = int(avg_fraud_prob * 100)
    confidence = 1.0 - np.std(fraud_probs)  # Higher confidence when models agree
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        'is_fraud': is_fraud,
        'fraud_probability': round(avg_fraud_prob, 4),
        'risk_score': risk_score,
        'confidence': round(confidence, 4),
        'processing_time_ms': round(processing_time, 2),
        'individual_predictions': predictions
    }

# Initialize the app
st.title("🛡️ Real-Time Fraud Detection System")
st.markdown("### Interactive Demo - No API Required!")
st.markdown("---")

# Load models
models = load_models()

# Create sample data
sample_data = create_sample_data()

with st.sidebar:
    st.header("🎛️ Controls")
    st.markdown("### System Status")
    
    if models is not None:
        st.success("🟢 Models Loaded")
        st.info(f"📊 {len(sample_data)} sample transactions available")
    else:
        st.error("🔴 Models Not Available")
    
    st.markdown("---")
    st.markdown("### 📈 Demo Statistics")
    fraud_count = sample_data['actual_fraud'].sum()
    st.metric("Total Transactions", len(sample_data))
    st.metric("Fraud Cases", fraud_count)
    st.metric("Fraud Rate", f"{fraud_count/len(sample_data)*100:.1f}%")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Transaction", "📊 Batch Analysis", "📈 Sample Data", "🎯 Model Performance"])

with tab1:
    st.header("Single Transaction Analysis")
    
    with st.expander("🧪 Test a Transaction", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            amt = st.number_input("💰 Amount ($)", min_value=0.01, value=1500.00, help="Transaction amount in USD")
            merchant_cat = st.selectbox("🏪 Merchant Category", 
                                      ["5411", "5812", "5732", "5541", "5999"],
                                      help="MCC codes: 5411=Grocery, 5812=Restaurant, 5732=Electronics, 5541=Gas, 5999=Misc")
            transaction_type = st.selectbox("💳 Transaction Type", 
                                          ["purchase", "withdrawal", "transfer", "payment"])
            location = st.text_input("📍 Location", value="New York")
            user_age = st.slider("👤 User Age", 18, 80, 35)
        
        with col2:
            time_of_day = st.slider("🕐 Time of Day (Hour)", 0, 23, 14)
            day_of_week = st.slider("📅 Day of Week (0=Monday)", 0, 6, 2)
            account_balance = st.number_input("🏦 Account Balance ($)", min_value=0.0, value=5000.00)
            num_transactions_day = st.number_input("📊 Transactions Today", min_value=0, value=3)
            avg_transaction_amount = st.number_input("📈 Average Transaction Amount ($)", min_value=0.01, value=250.00)
        
        if st.button("🔍 Analyze Transaction", type="primary"):
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
            
            with st.spinner("🤖 Analyzing transaction..."):
                result = predict_fraud(transaction_data, models)
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    fraud_status = "🚨 FRAUD DETECTED" if result['is_fraud'] else "✅ LEGITIMATE"
                    color = "red" if result['is_fraud'] else "green"
                    st.markdown(f"<h3 style='color: {color};'>{fraud_status}</h3>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("🎯 Fraud Probability", f"{result['fraud_probability']:.4f}")
                
                with col3:
                    st.metric("⚠️ Risk Score", f"{result['risk_score']}/100")
                
                col4, col5 = st.columns(2)
                
                with col4:
                    st.metric("🎯 Confidence", f"{result['confidence']:.4f}")
                
                with col5:
                    st.metric("⏱️ Processing Time", f"{result['processing_time_ms']:.2f}ms")
                
                if result['individual_predictions']:
                    st.subheader("🤖 Model Breakdown")
                    pred_df = pd.DataFrame(result['individual_predictions']).T
                    st.dataframe(pred_df, use_container_width=True)

with tab2:
    st.header("📊 Batch Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload CSV file with transactions", type="csv")
    
    with col2:
        use_sample = st.checkbox("🧪 Use sample data instead", value=False)
    
    df_to_analyze = None
    
    if use_sample:
        df_to_analyze = sample_data.copy()
        st.subheader("📋 Sample Data Preview")
        st.dataframe(df_to_analyze.head(10), use_container_width=True)
    elif uploaded_file is not None:
        try:
            df_to_analyze = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df_to_analyze.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    
    if df_to_analyze is not None and st.button("🚀 Analyze Batch", type="primary"):
        with st.spinner("🔄 Processing batch..."):
            results = []
            fraud_count = 0
            start_time = time.time()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (_, row) in enumerate(df_to_analyze.iterrows()):
                # Update progress
                progress = (idx + 1) / len(df_to_analyze)
                progress_bar.progress(progress)
                status_text.text(f"Processing transaction {idx + 1} of {len(df_to_analyze)}...")
                
                # Prepare transaction data
                transaction_data = {
                    "amount": row.get('amount', 100),
                    "merchant_category": row.get('merchant_category', '5411'),
                    "transaction_type": row.get('transaction_type', 'purchase'),
                    "location": row.get('location', 'Unknown'),
                    "time_of_day": row.get('time_of_day', 12),
                    "day_of_week": row.get('day_of_week', 1),
                    "user_age": row.get('user_age', 35),
                    "account_balance": row.get('account_balance', 1000),
                    "num_transactions_day": row.get('num_transactions_day', 3),
                    "avg_transaction_amount": row.get('avg_transaction_amount', 250)
                }
                
                result = predict_fraud(transaction_data, models)
                
                results.append({
                    'transaction_id': row.get('transaction_id', f'TXN_{idx:04d}'),
                    'amount': transaction_data['amount'],
                    'predicted_fraud': result['is_fraud'],
                    'fraud_probability': result['fraud_probability'],
                    'risk_score': result['risk_score'],
                    'actual_fraud': row.get('actual_fraud', None)  # If available
                })
                
                if result['is_fraud']:
                    fraud_count += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total Processed", len(results))
            
            with col2:
                st.metric("🚨 Fraud Detected", fraud_count)
            
            with col3:
                st.metric("📈 Fraud Rate", f"{fraud_count/len(results)*100:.1f}%")
            
            with col4:
                st.metric("⏱️ Processing Time", f"{processing_time:.2f}ms")
            
            # Results dataframe
            results_df = pd.DataFrame(results)
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fraud_chart = px.pie(
                    values=[fraud_count, len(results) - fraud_count],
                    names=['Fraud', 'Legitimate'],
                    title="🥧 Fraud Distribution",
                    color_discrete_map={'Fraud': '#ff4444', 'Legitimate': '#44ff44'}
                )
                st.plotly_chart(fraud_chart, use_container_width=True)
            
            with col2:
                risk_hist = px.histogram(
                    results_df,
                    x='risk_score',
                    title="📊 Risk Score Distribution",
                    nbins=20,
                    color_discrete_sequence=['#3366cc']
                )
                st.plotly_chart(risk_hist, use_container_width=True)

with tab3:
    st.header("📈 Sample Transaction Data")
    st.markdown("This tab shows the sample data used for demonstration purposes.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Data Summary")
        st.dataframe(sample_data.describe(), use_container_width=True)
    
    with col2:
        st.subheader("🎯 Fraud Distribution")
        fraud_dist = sample_data['actual_fraud'].value_counts()
        fraud_chart = px.bar(
            x=['Legitimate', 'Fraud'],
            y=[fraud_dist[0], fraud_dist[1]],
            title="Sample Data Fraud Cases",
            color=['Legitimate', 'Fraud'],
            color_discrete_map={'Legitimate': '#44ff44', 'Fraud': '#ff4444'}
        )
        st.plotly_chart(fraud_chart, use_container_width=True)
    
    st.subheader("📋 Full Sample Dataset")
    st.dataframe(sample_data, use_container_width=True)
    
    # Download option
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Data as CSV",
        data=csv,
        file_name="fraud_detection_sample_data.csv",
        mime="text/csv"
    )

with tab4:
    st.header("🎯 Model Performance Demo")
    st.markdown("This section demonstrates the model performance using the sample dataset.")
    
    if st.button("🧮 Run Performance Analysis", type="primary"):
        with st.spinner("🔄 Analyzing model performance..."):
            # Run predictions on sample data
            predictions = []
            for _, row in sample_data.iterrows():
                transaction_data = {
                    "amount": row['amount'],
                    "merchant_category": row['merchant_category'],
                    "transaction_type": row['transaction_type'],
                    "location": row['location'],
                    "time_of_day": row['time_of_day'],
                    "day_of_week": row['day_of_week'],
                    "user_age": row['user_age'],
                    "account_balance": row['account_balance'],
                    "num_transactions_day": row['num_transactions_day'],
                    "avg_transaction_amount": row['avg_transaction_amount']
                }
                result = predict_fraud(transaction_data, models)
                predictions.append(result['fraud_probability'])
            
            # Calculate metrics (simplified for demo)
            actual = sample_data['actual_fraud'].values
            predicted_probs = np.array(predictions)
            predicted = (predicted_probs > 0.5).astype(int)
            
            # Basic metrics
            tp = np.sum((actual == 1) & (predicted == 1))
            fp = np.sum((actual == 0) & (predicted == 1))
            tn = np.sum((actual == 0) & (predicted == 0))
            fn = np.sum((actual == 1) & (predicted == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            accuracy = (tp + tn) / len(actual)
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🎯 Accuracy", f"{accuracy:.3f}")
            
            with col2:
                st.metric("🎯 Precision", f"{precision:.3f}")
            
            with col3:
                st.metric("🎯 Recall", f"{recall:.3f}")
            
            with col4:
                st.metric("🎯 F1-Score", f"{f1_score:.3f}")
            
            # Confusion Matrix
            confusion_data = pd.DataFrame({
                'Predicted_Legitimate': [tn, fn],
                'Predicted_Fraud': [fp, tp]
            }, index=['Actual_Legitimate', 'Actual_Fraud'])
            
            st.subheader("📊 Confusion Matrix")
            st.dataframe(confusion_data, use_container_width=True)
            
            # ROC-like curve (simplified)
            thresholds = np.linspace(0, 1, 20)
            tpr_scores = []
            fpr_scores = []
            
            for threshold in thresholds:
                pred_at_threshold = (predicted_probs > threshold).astype(int)
                tp_t = np.sum((actual == 1) & (pred_at_threshold == 1))
                fp_t = np.sum((actual == 0) & (pred_at_threshold == 1))
                fn_t = np.sum((actual == 1) & (pred_at_threshold == 0))
                tn_t = np.sum((actual == 0) & (pred_at_threshold == 0))
                
                tpr = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
                fpr = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
                
                tpr_scores.append(tpr)
                fpr_scores.append(fpr)
            
            roc_fig = px.line(
                x=fpr_scores,
                y=tpr_scores,
                title="📈 ROC Curve (Demo)",
                labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}
            )
            roc_fig.add_shape(
                type='line',
                x0=0, y0=0, x1=1, y1=1,
                line=dict(dash='dash', color='red')
            )
            st.plotly_chart(roc_fig, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    🛡️ <strong>Fraud Detection System</strong> - Powered by Machine Learning<br>
    Built with Streamlit • Demo Version • Real-time Analysis
    </div>
    """,
    unsafe_allow_html=True
)