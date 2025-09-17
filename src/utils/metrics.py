"""
Metrics Collection and Monitoring
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from pathlib import Path
import time

class MetricsCollector:
    def __init__(self):
        self.metrics_file = Path("logs/metrics.json")
        self.start_time = datetime.now()
        self.predictions_today = 0
        self.fraud_detected_today = 0
        self.response_times = []
    
    async def initialize(self):
        self.metrics_file.parent.mkdir(exist_ok=True)
        if not self.metrics_file.exists():
            await self._save_metrics({
                "system_start": self.start_time.isoformat(),
                "total_predictions": 0,
                "fraud_detected": 0,
                "response_times": []
            })
    
    async def log_prediction(self, transaction: Dict, prediction: Dict, response_time: float):
        self.predictions_today += 1
        if prediction.get('is_fraud', False):
            self.fraud_detected_today += 1
        
        self.response_times.append(response_time)
        
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    async def log_batch_prediction(self, batch_size: int, response_time: float):
        self.predictions_today += batch_size
        self.response_times.append(response_time / batch_size)
    
    async def get_system_health(self) -> Dict[str, Any]:
        uptime = datetime.now() - self.start_time
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "status": "healthy",
            "api_uptime": str(uptime),
            "model_loaded": True,
            "total_predictions": self.predictions_today,
            "fraud_detected_today": self.fraud_detected_today,
            "avg_response_time_ms": round(avg_response_time, 2)
        }
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        return {
            "daily_stats": {
                "total_transactions": self.predictions_today,
                "fraud_detected": self.fraud_detected_today,
                "fraud_rate": round(self.fraud_detected_today / max(self.predictions_today, 1) * 100, 2)
            },
            "performance": {
                "avg_response_time_ms": round(sum(self.response_times) / len(self.response_times), 2) if self.response_times else 0,
                "uptime_hours": round((datetime.now() - self.start_time).total_seconds() / 3600, 2)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def cleanup(self):
        final_metrics = await self.get_analytics_summary()
        await self._save_metrics(final_metrics)
    
    async def _save_metrics(self, metrics: Dict):
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
