import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from pathlib import Path
import asyncio
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model_path = Path("data/models")
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.current_model = None

    async def retrain_model(self):
        logger.info("Retraining model...")

        # generate training data (TODO: use real data pipeline)
        X, y = self.make_training_data()
        # print(f"Generated {len(X)} samples")  # debug

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # train the model - random forest works pretty well
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        # check how it did
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        logger.info(f"Train acc: {train_score:.4f}, Test acc: {test_score:.4f}")

        # save it
        model_file = self.model_path / "real_world_ensemble_model.pkl"
        joblib.dump(model, model_file)

        self.current_model = model
        logger.info("Done retraining")

        return {
            "status": "success",
            "train_accuracy": train_score,
            "test_accuracy": test_score,
            "model_path": str(model_file)
        }

    def load_model(self):
        model_file = self.model_path / "real_world_ensemble_model.pkl"
        if model_file.exists():
            self.current_model = joblib.load(model_file)
            logger.info("Model loaded")
            return True
        else:
            logger.warning("No model file found")
            return False

    def get_model(self):
        if self.current_model is None:
            self.load_model()
        return self.current_model

    def make_training_data(self, n_samples=10000):
        # generate fake transaction data for training
        np.random.seed(42)
        
        # Create features
        features = []
        labels = []
        
        for i in range(n_samples):
            # Generate normal transaction (80% of data)
            if np.random.random() < 0.8:
                # Normal transaction patterns
                amount = np.random.lognormal(mean=3, sigma=1.5)
                time_of_day = np.random.normal(14, 4)  # Peak around 2 PM
                is_weekend = np.random.choice([0, 1], p=[0.7, 0.3])
                
                # Normal patterns
                account_balance = np.random.normal(5000, 2000)
                transaction_count = np.random.poisson(3)
                
                label = 0  # Not fraud
            else:
                # Fraudulent transaction patterns
                amount = np.random.lognormal(mean=5, sigma=2)  # Higher amounts
                time_of_day = np.random.choice([2, 3, 23, 24])  # Unusual hours
                is_weekend = np.random.choice([0, 1], p=[0.4, 0.6])  # More weekends
                
                # Suspicious patterns
                account_balance = np.random.normal(1000, 500)  # Lower balance
                transaction_count = np.random.poisson(8)  # More transactions
                
                label = 1  # Fraud
            
            # Normalize time_of_day
            time_of_day = max(0, min(23, time_of_day))
            
            # Create feature vector (15 features to match expected input)
            feature_vector = [
                amount / 1000,  # Normalized amount
                time_of_day / 24,  # Normalized time
                is_weekend,
                account_balance / 10000,  # Normalized balance
                transaction_count / 10,  # Normalized count
                np.sin(2 * np.pi * time_of_day / 24),  # Time sine
                np.cos(2 * np.pi * time_of_day / 24),  # Time cosine
                amount / (account_balance + 1),  # Amount to balance ratio
                np.random.normal(0, 0.1),  # Random feature 1
                np.random.normal(0, 0.1),  # Random feature 2
                np.random.normal(0, 0.1),  # Random feature 3
                np.random.normal(0, 0.1),  # Random feature 4
                np.random.normal(0, 0.1),  # Random feature 5
                np.random.normal(0, 0.1),  # Random feature 6
                np.random.normal(0, 0.1),  # Random feature 7
            ]
            
            features.append(feature_vector)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.current_model is None:
            return {"status": "No model loaded"}
        
        model_file = self.model_path / "real_world_ensemble_model.pkl"
        
        return {
            "model_type": type(self.current_model).__name__,
            "model_file": str(model_file),
            "file_exists": model_file.exists(),
            "n_estimators": getattr(self.current_model, 'n_estimators', 'N/A'),
            "max_depth": getattr(self.current_model, 'max_depth', 'N/A'),
        }
