from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import uvicorn
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.ml.predictor import FraudPredictor
from src.ml.model_manager import ModelManager
from src.utils.logger import setup_logger
from src.utils.metrics import MetricsCollector

logger = setup_logger("fraud_api")

app = FastAPI(title="Fraud Detector", version="1.0")
# TODO: add better error handling for edge cases

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init the main components
model_manager = ModelManager()
predictor = FraudPredictor()
metrics = MetricsCollector()

class TransactionRequest(BaseModel):
    amount: float
    merchant_category: str
    transaction_type: str
    location: str
    time_of_day: int
    day_of_week: int
    user_age: int
    account_balance: float
    num_transactions_day: int
    avg_transaction_amount: float

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    risk_score: int
    confidence: float
    model_used: str
    processing_time_ms: float
    timestamp: str

class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]

class ModelMetrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    
class SystemHealth(BaseModel):
    status: str
    api_uptime: str
    model_loaded: bool
    total_predictions: int
    fraud_detected_today: int
    avg_response_time_ms: float

@app.get("/")
async def root():
    return {
        "message": "Fraud detector running",
        "version": "1.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    health_data = await metrics.get_system_health()
    return health_data

@app.post("/predict")
async def predict_fraud(transaction: TransactionRequest, background_tasks: BackgroundTasks):
    start_time = datetime.now()
    
    # just run the prediction
    transaction_data = pd.DataFrame([transaction.dict()])
    prediction = await predictor.predict(transaction_data)
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    # print(f"Debug: prediction took {processing_time}ms")  # keeping this for now
    
    # log it in background
    background_tasks.add_task(
        metrics.log_prediction,
        transaction.dict(),
        prediction,
        processing_time
    )
    
    return {
        "is_fraud": prediction['is_fraud'],
        "fraud_probability": prediction['probability'],
        "risk_score": prediction['risk_score'],
        "confidence": prediction['confidence'],
        "model_used": prediction['model'],
        "processing_time_ms": round(processing_time, 2),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/batch")
async def predict_fraud_batch(batch_request: BatchTransactionRequest, background_tasks: BackgroundTasks):
    start_time = datetime.now()
    
    # process all transactions
    transactions_data = pd.DataFrame([t.dict() for t in batch_request.transactions])
    predictions = await predictor.predict_batch(transactions_data)
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    background_tasks.add_task(
        metrics.log_batch_prediction,
        len(batch_request.transactions),
        processing_time
    )
    
    return {
        "predictions": predictions,
        "total_processed": len(batch_request.transactions),
        "processing_time_ms": round(processing_time, 2),
        "fraud_detected": sum(1 for p in predictions if p['is_fraud']),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models/metrics")
async def get_model_metrics():
    metrics_data = await model_manager.get_model_metrics()
    return metrics_data

@app.post("/models/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    background_tasks.add_task(model_manager.retrain_model)
    return {
        "message": "Started retraining the model",
        "status": "in_progress",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/analytics/summary")
async def get_analytics_summary():
    summary = await metrics.get_analytics_summary()
    return summary

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up...")
    await predictor.load_model()
    await metrics.initialize()
    logger.info("Ready to detect fraud!")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    await metrics.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
