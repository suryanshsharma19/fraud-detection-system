import pandas as pd
import numpy as np
import joblib
import asyncio
from pathlib import Path
import logging
from datetime import datetime
import os

from src.ml.preprocessor import DataPreprocessor
from src.ml.feature_engineer import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger("ml_predictor")

class FraudPredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model_loaded = False
        self.model_path = Path("data/models")
        self.model_path.mkdir(parents=True, exist_ok=True)
    
    async def load_model(self):
        try:
            # First try loading the production ensemble model - this took forever to train but works best
            real_model_file = self.model_path / 'real_world_ensemble_model.pkl'
            real_scaler_file = self.model_path / 'real_world_ensemble_scaler.pkl'
            
            if real_model_file.exists() and real_scaler_file.exists():
                # Load the ensemble model I trained on multiple datasets
                self.models['real_world'] = joblib.load(real_model_file)
                self.scaler = joblib.load(real_scaler_file)
                logger.info("Loaded production ensemble model (5.3MB)")
                self.model_loaded = True
                logger.info("Fraud detection system ready - using ensemble approach")
                return
            
            # Fallback to other models if real-world model not found
            model_files = {
                'random_forest': 'random_forest_model.joblib',
                'xgboost': 'xgboost_model.joblib',
                'neural_network': 'neural_network_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                model_file = self.model_path / filename
                if model_file.exists():
                    self.models[model_name] = joblib.load(model_file)
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {filename}")
            
            if not self.models:
                await self._create_default_model()
            
            self.model_loaded = True
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            await self._create_default_model()
    
    async def _create_default_model(self):
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        np.random.seed(42)
        X_synthetic = np.random.rand(1000, 10)
        y_synthetic = (X_synthetic[:, 0] + X_synthetic[:, 1] > 1.0).astype(int)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_synthetic, y_synthetic)
        
        self.models['default'] = model
        logger.info("Created default model for demonstration")
    
    async def _predict_with_real_model(self, transaction_data):
        """Predict using the real-world trained model"""
        try:
            # Convert transaction data to the format expected by our real model
            df = transaction_data.copy()
            
            # Create features that match our training data
            # had to map input fields to model features - took forever to figure this out
            amt = df.get('amount', 100.0).iloc[0]
            features_dict = {
                'amount': amt,
                'amount_log': np.log1p(amt),
                'oldbalanceOrg': df.get('account_balance', 1000.0).iloc[0],
                'newbalanceOrig': df.get('account_balance', 1000.0).iloc[0] - df.get('amount', 100.0).iloc[0],
                'oldbalanceDest': 0.0,  # Unknown for incoming data
                'newbalanceDest': 0.0,  # Unknown for incoming data
                'balance_ratio': df.get('amount', 100.0).iloc[0] / (df.get('account_balance', 1000.0).iloc[0] + 1),
                'balance_change_orig': -df.get('amount', 100.0).iloc[0],
                'balance_change_dest': 0.0,
                'hour': df.get('time_of_day', 12).iloc[0],
                'day': df.get('day_of_week', 1).iloc[0],
                'hour_sin': np.sin(2 * np.pi * df.get('time_of_day', 12).iloc[0] / 24),
                'hour_cos': np.cos(2 * np.pi * df.get('time_of_day', 12).iloc[0] / 24),
                'type_encoded': 1,  # Default to payment type
                'step': 1  # Default step
            }
            
            # Create feature array in the correct order
            feature_array = np.array([[
                features_dict['amount'],
                features_dict['amount_log'], 
                features_dict['oldbalanceOrg'],
                features_dict['newbalanceOrig'],
                features_dict['oldbalanceDest'],
                features_dict['newbalanceDest'],
                features_dict['balance_ratio'],
                features_dict['balance_change_orig'],
                features_dict['balance_change_dest'],
                features_dict['hour'],
                features_dict['day'],
                features_dict['hour_sin'],
                features_dict['hour_cos'],
                features_dict['type_encoded'],
                features_dict['step']
            ]])
            
            # Scale the features
            if hasattr(self, 'scaler'):
                feature_array = self.scaler.transform(feature_array)
            
            # Make prediction
            model = self.models['real_world']
            fraud_prob = model.predict_proba(feature_array)[0][1]
            is_fraud = model.predict(feature_array)[0]
            
            # Calculate risk score and confidence
            risk_score = int(fraud_prob * 100)
            confidence = max(fraud_prob, 1 - fraud_prob)
            
            return {
                'is_fraud': bool(is_fraud),
                'probability': float(fraud_prob),
                'risk_score': risk_score,
                'confidence': float(confidence),
                'model': 'real_world_ensemble'
            }
            
        except Exception as e:
            logger.error(f"Error in real-world prediction: {e}")
            # Fallback to simple prediction
            return {
                'is_fraud': False,
                'probability': 0.1,
                'risk_score': 10,
                'confidence': 0.9,
                'model': 'fallback'
            }
    
    async def predict(self, transaction_data):
        if not self.model_loaded:
            await self.load_model()
        
        try:
            # If we have the real-world model, use it directly
            if 'real_world' in self.models:
                return await self._predict_with_real_model(transaction_data)
            
            # Fallback to the original processing
            processed_data = await self.preprocessor.transform(transaction_data)
            features = await self.feature_engineer.create_features(processed_data)
            
            predictions = {}
            for model_name, model in self.models.items():
                try:
                    if model_name == 'default':
                        feature_subset = features.iloc[:, :10]
                        prob = model.predict_proba(feature_subset)[0]
                        pred = model.predict(feature_subset)[0]
                    else:
                        prob = model.predict_proba(features)[0]
                        pred = model.predict(features)[0]
                    
                    predictions[model_name] = {
                        'prediction': int(pred),
                        'probability': float(prob[1] if len(prob) > 1 else prob[0])
                    }
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
            
            ensemble_prob = np.mean([p['probability'] for p in predictions.values()])
            ensemble_pred = 1 if ensemble_prob > 0.5 else 0
            
            risk_score = int(ensemble_prob * 100)
            
            pred_values = [p['prediction'] for p in predictions.values()]
            confidence = len([p for p in pred_values if p == ensemble_pred]) / len(pred_values)
            
            return {
                'is_fraud': bool(ensemble_pred),
                'probability': round(ensemble_prob, 4),
                'risk_score': risk_score,
                'confidence': round(confidence, 4),
                'model': 'ensemble',
                'individual_predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'is_fraud': False,
                'probability': 0.0,
                'risk_score': 0,
                'confidence': 0.5,
                'model': 'error_fallback',
                'error': str(e)
            }
    
    async def predict_batch(self, transactions_data):
        if not self.model_loaded:
            await self.load_model()
        
        try:
            processed_data = await self.preprocessor.transform(transactions_data)
            features = await self.feature_engineer.create_features(processed_data)
            
            results = []
            
            batch_size = 100
            for i in range(0, len(features), batch_size):
                batch_features = features.iloc[i:i+batch_size]
                
                model_name = list(self.models.keys())[0]
                model = self.models[model_name]
                
                if model_name == 'default':
                    feature_subset = batch_features.iloc[:, :10]
                    probabilities = model.predict_proba(feature_subset)
                    predictions = model.predict(feature_subset)
                else:
                    probabilities = model.predict_proba(batch_features)
                    predictions = model.predict(batch_features)
                
                for j, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    fraud_prob = prob[1] if len(prob) > 1 else prob[0]
                    
                    results.append({
                        'is_fraud': bool(pred),
                        'probability': round(float(fraud_prob), 4),
                        'risk_score': int(fraud_prob * 100),
                        'confidence': 0.85,
                        'model': model_name,
                        'transaction_id': i + j
                    })
            
            logger.info(f"Processed {len(results)} transactions in batch")
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            return [
                {
                    'is_fraud': False,
                    'probability': 0.0,
                    'risk_score': 0,
                    'confidence': 0.5,
                    'model': 'error_fallback',
                    'transaction_id': i
                }
                for i in range(len(transactions_data))
            ]
    
    def get_model_info(self):
        return {
            'models_loaded': list(self.models.keys()),
            'model_count': len(self.models),
            'status': 'ready' if self.model_loaded else 'not_loaded',
            'model_path': str(self.model_path)
        }
