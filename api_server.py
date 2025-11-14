"""
PRODUCTION API SERVER
FastAPI wrapper for semantic space trading predictions

Endpoints:
- POST /predict: Get predictions for all tickers
- GET /health: Health check
- GET /status: Model status and metrics

Author: RD-Agent Research Team
Date: 2025-11-14
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import torch
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import uvicorn
import os
from functools import lru_cache

from production_semantic_trader import ProductionSemanticTrader

# API Keys (in production, use environment variables)
API_KEYS = {
    "demo_key_12345": {"tier": "starter", "name": "Demo Account"},
    # Add real API keys from Stripe webhook or database
}

app = FastAPI(
    title="Semantic Space Trading API",
    description="Real-time trading predictions using semantic space neural networks",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class PredictionRequest(BaseModel):
    tickers: Optional[List[str]] = Field(
        default=None,
        description="List of tickers to predict (if None, uses default 23)"
    )
    confidence_threshold: Optional[float] = Field(
        default=0.001,
        description="Minimum predicted return threshold (default 0.1%)"
    )

class PredictionResponse(BaseModel):
    timestamp: str
    predictions: Dict[str, float]
    top_longs: List[Dict[str, float]]
    top_shorts: List[Dict[str, float]]
    model_version: str
    model_ic: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

# Global trader instance
trader = None

def verify_api_key(x_api_key: str = Header(...)):
    """Verify API key from header."""
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return API_KEYS[x_api_key]


@app.on_event("startup")
async def startup_event():
    """Initialize trader on startup."""
    global trader

    print("Loading semantic network...")

    TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        'JPM', 'BAC', 'GS', 'MS',
        'WMT', 'HD', 'MCD', 'NKE',
        'JNJ', 'UNH', 'PFE',
        'XOM', 'CVX',
        'SPY', 'QQQ', 'IWM',
    ]

    trader = ProductionSemanticTrader(
        model_path='semantic_network_best.pt',
        tickers=TICKERS,
        initial_capital=100000.0,
        max_position_size=0.10,
        transaction_cost_bps=2.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )

    print("✓ API server ready")


@app.get("/", response_model=Dict)
async def root():
    """API root endpoint."""
    return {
        "name": "Semantic Space Trading API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if trader is not None else "initializing",
        model_loaded=trader is not None,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def get_predictions(
    request: PredictionRequest,
    account: dict = Depends(verify_api_key)
):
    """
    Get trading predictions.

    Returns predicted returns for all tickers, along with top long/short recommendations.
    """
    if trader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Fetch live data
        market_data = trader.fetch_live_data(lookback_days=30)

        if len(market_data) == 0:
            raise HTTPException(status_code=503, detail="Market data unavailable")

        # Get predictions
        predictions = trader.get_predictions(market_data)

        # Filter by confidence threshold
        filtered_predictions = {
            ticker: pred
            for ticker, pred in predictions.items()
            if abs(pred) >= request.confidence_threshold
        }

        # Get top longs
        top_longs = sorted(
            [(ticker, pred) for ticker, pred in filtered_predictions.items() if pred > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Get top shorts
        top_shorts = sorted(
            [(ticker, pred) for ticker, pred in filtered_predictions.items() if pred < 0],
            key=lambda x: x[1]
        )[:5]

        return PredictionResponse(
            timestamp=datetime.now().isoformat(),
            predictions=predictions,
            top_longs=[{"ticker": t, "predicted_return": p} for t, p in top_longs],
            top_shorts=[{"ticker": t, "predicted_return": p} for t, p in top_shorts],
            model_version="semantic_v1.0",
            model_ic=0.0152  # From validation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/tickers")
async def get_tickers(account: dict = Depends(verify_api_key)):
    """Get list of supported tickers."""
    if trader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "tickers": trader.tickers,
        "count": len(trader.tickers)
    }


@app.get("/performance")
async def get_performance(account: dict = Depends(verify_api_key)):
    """Get model performance metrics."""
    return {
        "validation_ic": 0.0152,
        "validation_sharpe": 0.31,
        "model_architecture": "Semantic Space Transformer",
        "parameters": 3780000,
        "training_epochs": 181,
        "features": [
            "Returns (log)",
            "Volume (normalized)",
            "Pairwise correlations",
            "Multi-head attention",
            "Semantic embeddings"
        ]
    }


if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
