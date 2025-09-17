# cleans up the raw transaction data before feeding to ML model

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
import asyncio
from typing import Dict, List, Any
import logging

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders = {}  # for categorical stuff
        self.is_fitted = False

    async def transform(self, data):
        df = data.copy()

        # add some extra features if we don't have enough
        if len(df.columns) < 15:
            df = self.add_extra_features(df)

        # encode text columns
        text_cols = ['merchant_category', 'transaction_type', 'location']
        for col in text_cols:
            if col in df.columns:
                df[f'{col}_encoded'] = self.encode_text(df[col], col)

        # normalize numbers
        num_cols = [
            'amount', 'user_age', 'account_balance',
            'num_transactions_day', 'avg_transaction_amount'
        ]

        for col in num_cols:
            if col in df.columns:
                df[f'{col}_normalized'] = self.normalize_column(df[col])

        # get all the features we made
        feature_cols = [col for col in df.columns if
                       col.endswith('_encoded') or col.endswith('_normalized') or
                       col in ['time_of_day', 'day_of_week']]

        # pad to 15 features if needed
        while len(feature_cols) < 15:
            feature_cols.append(f'extra_feature_{len(feature_cols)}')
            df[f'extra_feature_{len(feature_cols)-1}'] = np.random.rand(len(df))

        return df[feature_cols]

    def add_extra_features(self, df):
        # add some derived features that might help
        df['amount_log'] = np.log1p(df.get('amount', 100))
        df['amount_zscore'] = (df.get('amount', 100) - 500) / 1000
        
        df['is_weekend'] = df.get('day_of_week', 1).apply(lambda x: 1 if x in [5, 6] else 0)
        df['is_night'] = df.get('time_of_day', 12).apply(lambda x: 1 if x < 6 or x > 22 else 0)
        
        df['balance_ratio'] = df.get('amount', 100) / (df.get('account_balance', 1000) + 1)
        df['txn_frequency'] = df.get('num_transactions_day', 1) / 10
        
        df['big_amount'] = (df.get('amount', 100) > df.get('avg_transaction_amount', 250) * 3).astype(int)
        df['weird_time'] = ((df.get('time_of_day', 12) < 6) | (df.get('time_of_day', 12) > 23)).astype(int)
        
        return df
    
    def encode_text(self, series, col_name):
        # convert text to numbers for ML
        if col_name not in self.encoders:
            self.encoders[col_name] = LabelEncoder()
            cats = series.unique().tolist()
            if len(cats) == 0:
                cats = ['unknown']
            self.encoders[col_name].fit(cats)

        known_cats = set(self.encoders[col_name].classes_)
        series_clean = series.apply(lambda x: x if x in known_cats else 'unknown')

        if 'unknown' not in known_cats:
            current_cats = self.encoders[col_name].classes_.tolist()
            current_cats.append('unknown')
            self.encoders[col_name].classes_ = np.array(current_cats)

        return pd.Series(self.encoders[col_name].transform(series_clean))

    def normalize_column(self, series):
        # simple normalization
        return (series - series.mean()) / (series.std() + 1e-8)