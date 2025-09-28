import os
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware   # ✅ Added CORS
from pydantic import BaseModel, Field

# Optional: OpenAI rationale
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

app = FastAPI(title="KagamiFi – Simple Planner")

# ✅ CORS middleware (allow your frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # ⚠️ in prod, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LLAMA_TIMEOUT = 25.0

# ✅ Use contract-address mapping for ERC20 tokens
TOKEN_MAP = {
    "ETH": "ethereum:0xC02aaA39b223FE8D0A0E5C4F27eAD9083C756Cc2",  # WETH
    "USDC": "ethereum:0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "USDT": "ethereum:0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "DAI": "ethereum:0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "BTC": "coingecko:bitcoin"  # Non-ERC20 fallback
}

def _token_id(symbol: str) -> str:
    sym = symbol.upper()
    return TOKEN_MAP.get(sym, f"coingecko:{symbol.lower()}")

# ---------- Price fetching ----------
async def get_current_price(symbol: str) -> Optional[float]:
    url = f"https://coins.llama.fi/prices/current/{_token_id(symbol)}"
    async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        return data.get("coins", {}).get(_token_id(symbol), {}).get("price")

async def get_historical_prices(symbol: str, days: int = 120) -> List[Dict[str, float]]:
    start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
    url = f"https://coins.llama.fi/chart/{_token_id(symbol)}?start={start}"
    async with httpx.AsyncClient(timeout=LLAMA_TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return data.get("prices", [])

# ---------- Indicators ----------
def rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    diffs = np.diff(np.array(prices))
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def annualized_vol(prices: List[float]) -> Optional[float]:
    if len(prices) < 2:
        return None
    rets = np.diff(np.log(np.array(prices)))
    daily_vol = np.std(rets) * math.sqrt(365)
    return float(daily_vol)

def safe_float(x: float | str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0

def normalize_amounts(total_usd: float) -> Dict[str, float]:
    return {
        "LP": round(total_usd * 0.60, 2),
        "TWAP": round(total_usd * 0.30, 2),
        "LIMITS": round(total_usd * 0.10, 2),
    }

def days_list(n: int) -> List[str]:
    base = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return [(base + timedelta(days=i)).isoformat() for i in range(n)]

# ---------- Models ----------
class Range(BaseModel):
    lower: float
    upper: float

class BlockLimitOrderParams(BaseModel):
    grid: List[List[float]]

class Block(BaseModel):
    kind: str
    protocol: Optional[str] = None
    pair: Optional[str] = None
    feeTierBps: Optional[int] = None
    range: Optional[Range] = None
    hook: Optional[str] = None
    params: Optional[BlockLimitOrderParams] = None
    windowSecs: Optional[int] = None
    maxSlippageBps: Optional[int] = None

class RiskControls(BaseModel):
    maxSlippageBps: Optional[int] = 30
    stopLossPct: Optional[float] = -6

class CopyTrading(BaseModel):
    feeFollowPct: Optional[float] = 5
    performanceFeePct: Optional[float] = 10

class TradingAlgoIn(BaseModel):
    name: str
    type: str
    riskLevel: str
    timeHorizon: str
    tokens: List[str]
    amount: float | str
    slippage: float | str
    gasPrice: float | str
    conditions: List[str] = []
    blocks: List[Block] = Field(default_factory=list)
    riskControls: Optional[RiskControls] = None
    copyTrading: Optional[CopyTrading] = None

# ---------- Planner (same as before) ----------
# ... keep the build_90d_plan function here (unchanged) ...

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.post("/plan/90d")
async def plan_90d(payload: TradingAlgoIn):
    return await build_90d_plan(payload)
