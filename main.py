import os
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Optional: OpenAI rationale
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

app = FastAPI(title="KagamiFi – Simple Planner")

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

# ---------- Planner ----------
async def build_90d_plan(payload: TradingAlgoIn) -> Dict[str, Any]:
    if not payload.tokens:
        raise HTTPException(400, "tokens[] required")

    # Pick primary (first non-stablecoin if possible)
    primary = next((t for t in payload.tokens if t.upper() not in ("USDC", "USDT", "DAI")), payload.tokens[0])

    hist = await get_historical_prices(primary, days=120)
    cur_price = await get_current_price(primary)
    if cur_price is None:
        raise HTTPException(502, f"No current price for {primary}")

    prices = [float(x["price"]) for x in hist] if hist else []
    if not prices:
        prices = [cur_price]

    px_now = cur_price
    rsi14 = rsi(prices, 14)
    volA = annualized_vol(prices) if len(prices) > 1 else None

    total_usd = safe_float(payload.amount)
    allocations = normalize_amounts(total_usd)

    has_lp = any(b.kind.upper() == "AMM_LP" for b in payload.blocks)
    has_twap = any(b.kind.upper() == "TWAP" for b in payload.blocks)
    has_limits = any(b.kind.upper() == "LIMIT_ORDER" for b in payload.blocks)
    if not (has_lp or has_twap or has_limits):
        has_twap = True

    days90 = days_list(90)

    twap_plan: List[Dict[str, Any]] = []
    if has_twap:
        daily_usd = round(allocations["TWAP"] / 90.0, 2)
        default_slip = 25
        for b in payload.blocks:
            if b.kind.upper() == "TWAP" and b.maxSlippageBps is not None:
                default_slip = b.maxSlippageBps
        for day in days90:
            twap_plan.append({
                "date": day,
                "action": "BUY_TWAP",
                "symbol": primary.upper(),
                "usd_amount": daily_usd,
                "max_slippage_bps": default_slip
            })

    lp_plan: List[Dict[str, Any]] = []
    if has_lp:
        lower, upper, fee_bps = None, None, 3000
        proto = "UniswapV3"
        pair = f"{primary.upper()}/" + ("USDC" if "USDC" in [t.upper() for t in payload.tokens] else payload.tokens[-1].upper())
        for b in payload.blocks:
            if b.kind.upper() == "AMM_LP":
                if b.range:
                    lower, upper = b.range.lower, b.range.upper
                if b.feeTierBps: fee_bps = b.feeTierBps
                if b.protocol: proto = b.protocol
                if b.pair: pair = b.pair
        if lower is None or upper is None:
            width = 0.25
            lower, upper = px_now * (1 - width), px_now * (1 + width)
        step_days = 7 if not (volA and volA > 1.0) else 5
        for i, day in enumerate(days90):
            if i % step_days == 0:
                lp_plan.append({
                    "date": day,
                    "action": "LP_CHECK",
                    "protocol": proto,
                    "pair": pair,
                    "fee_tier_bps": fee_bps,
                    "current_price_hint": round(px_now, 4),
                    "active_range": {"lower": lower, "upper": upper},
                    "rebalance_rule": "If price exits range, rebalance"
                })

    grid_plan: List[Dict[str, Any]] = []
    if has_limits:
        grids = []
        for b in payload.blocks:
            if b.kind.upper() == "LIMIT_ORDER" and b.params and b.params.grid:
                grids = b.params.grid
                break
        total_pct = sum(row[1] for row in grids) if grids else 0
        for price_level, pct in grids:
            share = allocations["LIMITS"] * (pct / total_pct) if total_pct > 0 else 0.0
            grid_plan.append({
                "when": "ANYTIME_UNTIL_FILLED",
                "action": "PLACE_LIMIT",
                "symbol": primary.upper(),
                "limit_price": float(price_level),
                "usd_amount": round(share, 2),
                "note": f"{pct}% of LIMITS"
            })

    risk = {
        "max_slippage_bps": (payload.riskControls.maxSlippageBps
                             if payload.riskControls and payload.riskControls.maxSlippageBps is not None else 30),
        "stop_loss_pct": (payload.riskControls.stopLossPct
                          if payload.riskControls and payload.riskControls.stopLossPct is not None else -6),
        "rsi_now": None if rsi14 is None else round(rsi14, 2),
        "ann_vol_est": None if volA is None else round(volA, 3)
    }

    header = {
        "algo_name": payload.name,
        "primary_symbol": primary.upper(),
        "now_price": round(px_now, 4),
        "allocations_usd": allocations,
        "signals_snapshot": {
            "RSI14": risk["rsi_now"],
            "ann_vol": risk["ann_vol_est"]
        }
    }

    plan = {
        "header": header,
        "twap_schedule": twap_plan,
        "lp_maintenance": lp_plan,
        "limit_orders": grid_plan,
        "risk_controls": risk,
        "notes": {
            "conditions": payload.conditions,
            "assumptions": [
                "Using current price fallback when history missing",
                "TWAP spreads across 90 days",
                "LP rebalancing periodic",
                "Limits stand until filled"
            ]
        }
    }

    if openai_client:
        try:
            prompt = (
                f"Given a {payload.type} strategy '{payload.name}' on {primary.upper()} with budget "
                f"${total_usd:.2f}, RSI ~ {risk['rsi_now']}, vol ~ {risk['ann_vol_est']}, "
                "explain why using TWAP, LP, and limits, and how risk controls help."
            )
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            plan["ai_rationale"] = resp.choices[0].message.content
        except Exception as e:
            plan["ai_rationale"] = f"(AI rationale error: {e})"

    return plan

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.post("/plan/90d")
async def plan_90d(payload: TradingAlgoIn):
    return await build_90d_plan(payload)
