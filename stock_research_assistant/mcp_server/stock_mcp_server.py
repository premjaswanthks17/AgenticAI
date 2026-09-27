"""
mcp_server/stock_mcp_server.py
──────────────────────────────
Custom MCP Server for the Stock Research & Portfolio Assistant.

Exposes THREE tools (mandatory requirement):
  1. get_price        – current & historical OHLCV prices via yfinance
  2. get_financials   – key ratios and P&L summary via yfinance
  3. calc_portfolio_risk – portfolio-level volatility, beta & concentration

Run standalone:
    python mcp_server/stock_mcp_server.py

The LangGraph orchestrator connects to this server over stdio transport
using langchain-mcp-adapters.
"""

import json
import sys
from datetime import datetime, date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

# ── FastMCP app ────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="StockResearchMCP",
    instructions=(
        "You are a financial data retrieval server. "
        "Return raw data only – no buy/sell recommendations."
    ),
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _safe_round(value: Any, decimals: int = 2) -> Any:
    """Round floats safely, return None for NaN/inf."""
    try:
        v = float(value)
        if not np.isfinite(v):
            return None
        return round(v, decimals)
    except (TypeError, ValueError):
        return None


def _ticker_obj(ticker: str) -> yf.Ticker:
    """Return a yfinance Ticker, appending .NS for Indian equities if needed."""
    ticker = ticker.strip().upper()
    if "." not in ticker and not ticker.startswith("^"):
        # Assume Indian NSE ticker if no exchange suffix given
        ticker = ticker + ".NS"
    return yf.Ticker(ticker)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 – get_price
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_price(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Return current price and historical OHLCV data for a stock ticker.

    Args:
        ticker:   Stock symbol, e.g. 'INFY', 'TCS', 'RELIANCE', or 'AAPL'.
                  Indian NSE tickers are recognised automatically (suffix .NS added).
        period:   Lookback window – valid values: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y.
        interval: Bar interval  – valid values: 1m, 5m, 15m, 1h, 1d, 1wk, 1mo.

    Returns:
        JSON string with current price snapshot and last 10 bars of OHLCV.
    """
    try:
        t = _ticker_obj(ticker)
        hist = t.history(period=period, interval=interval)

        if hist.empty:
            return json.dumps({"error": f"No price data found for '{ticker}'"})

        info = t.fast_info
        current_price = _safe_round(info.last_price) if hasattr(info, "last_price") else None
        prev_close = _safe_round(info.previous_close) if hasattr(info, "previous_close") else None

        pct_change = None
        if current_price and prev_close and prev_close != 0:
            pct_change = _safe_round((current_price - prev_close) / prev_close * 100)

        # Last 10 bars summary
        bars = []
        for idx_date, row in hist.tail(10).iterrows():
            bars.append({
                "date": str(idx_date.date() if hasattr(idx_date, "date") else idx_date),
                "open": _safe_round(row["Open"]),
                "high": _safe_round(row["High"]),
                "low": _safe_round(row["Low"]),
                "close": _safe_round(row["Close"]),
                "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else None,
            })

        # 52-week range
        hist_1y = t.history(period="1y", interval="1d")
        week52_high = _safe_round(hist_1y["High"].max()) if not hist_1y.empty else None
        week52_low = _safe_round(hist_1y["Low"].min()) if not hist_1y.empty else None

        result = {
            "ticker": ticker.upper(),
            "as_of": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "current_price": current_price,
            "previous_close": prev_close,
            "change_pct": pct_change,
            "week_52_high": week52_high,
            "week_52_low": week52_low,
            "currency": getattr(info, "currency", "INR"),
            "period_requested": period,
            "interval": interval,
            "ohlcv_last_10_bars": bars,
        }
        return json.dumps(result, default=str)

    except Exception as exc:
        return json.dumps({"error": str(exc), "ticker": ticker})


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 – get_financials
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_financials(ticker: str) -> str:
    """
    Return key financial ratios, valuation metrics and income-statement highlights.

    Args:
        ticker: Stock symbol, e.g. 'INFY', 'TCS', 'RELIANCE'.

    Returns:
        JSON string with valuation ratios (P/E, P/B, EV/EBITDA), profitability
        (ROE, ROA, profit margin), debt metrics (D/E, interest coverage),
        dividend yield, and the last 4 quarters of revenue & net income.
    """
    try:
        t = _ticker_obj(ticker)
        info = t.info  # full metadata dict

        # ── Valuation ──────────────────────────────────────────────────────────
        valuation = {
            "pe_ratio_ttm": _safe_round(info.get("trailingPE")),
            "forward_pe": _safe_round(info.get("forwardPE")),
            "price_to_book": _safe_round(info.get("priceToBook")),
            "ev_to_ebitda": _safe_round(info.get("enterpriseToEbitda")),
            "price_to_sales_ttm": _safe_round(info.get("priceToSalesTrailing12Months")),
            "market_cap": info.get("marketCap"),
            "enterprise_value": info.get("enterpriseValue"),
        }

        # ── Profitability ──────────────────────────────────────────────────────
        profitability = {
            "roe_pct": _safe_round(
                info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None
            ),
            "roa_pct": _safe_round(
                info.get("returnOnAssets", 0) * 100 if info.get("returnOnAssets") else None
            ),
            "gross_margin_pct": _safe_round(
                info.get("grossMargins", 0) * 100 if info.get("grossMargins") else None
            ),
            "operating_margin_pct": _safe_round(
                info.get("operatingMargins", 0) * 100 if info.get("operatingMargins") else None
            ),
            "net_profit_margin_pct": _safe_round(
                info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None
            ),
            "revenue_growth_yoy_pct": _safe_round(
                info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None
            ),
            "earnings_growth_yoy_pct": _safe_round(
                info.get("earningsGrowth", 0) * 100 if info.get("earningsGrowth") else None
            ),
        }

        # ── Debt & Liquidity ───────────────────────────────────────────────────
        debt = {
            "debt_to_equity": _safe_round(info.get("debtToEquity")),
            "current_ratio": _safe_round(info.get("currentRatio")),
            "quick_ratio": _safe_round(info.get("quickRatio")),
            "total_debt": info.get("totalDebt"),
            "free_cashflow": info.get("freeCashflow"),
        }

        # ── Dividends ──────────────────────────────────────────────────────────
        dividends = {
            "dividend_yield_pct": _safe_round(
                info.get("dividendYield", 0) * 100 if info.get("dividendYield") else None
            ),
            "payout_ratio_pct": _safe_round(
                info.get("payoutRatio", 0) * 100 if info.get("payoutRatio") else None
            ),
        }

        # ── Quarterly revenue & net income ────────────────────────────────────
        try:
            q_financials = t.quarterly_financials
            quarterly = []
            if q_financials is not None and not q_financials.empty:
                for col in q_financials.columns[:4]:
                    quarter_data = {"period": str(col.date() if hasattr(col, "date") else col)}
                    if "Total Revenue" in q_financials.index:
                        quarter_data["revenue"] = _safe_round(
                            q_financials.loc["Total Revenue", col] / 1e7, 2
                        )
                        quarter_data["revenue_unit"] = "Crore INR"
                    if "Net Income" in q_financials.index:
                        quarter_data["net_income"] = _safe_round(
                            q_financials.loc["Net Income", col] / 1e7, 2
                        )
                        quarter_data["net_income_unit"] = "Crore INR"
                    quarterly.append(quarter_data)
        except Exception:
            quarterly = []

        result = {
            "ticker": ticker.upper(),
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
            "valuation": valuation,
            "profitability": profitability,
            "debt_and_liquidity": debt,
            "dividends": dividends,
            "quarterly_financials_last_4": quarterly,
            "business_summary": (info.get("longBusinessSummary", "") or "")[:500],
        }
        return json.dumps(result, default=str)

    except Exception as exc:
        return json.dumps({"error": str(exc), "ticker": ticker})


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 – calc_portfolio_risk
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def calc_portfolio_risk(
    holdings: list,
    weights: list,
    period: str = "1y",
    benchmark_ticker: str = "^NSEI",
) -> str:
    """
    Compute portfolio-level risk metrics: volatility, beta, Sharpe estimate
    and concentration (Herfindahl–Hirschman Index).

    Args:
        holdings:         List of ticker symbols, e.g. ["TCS", "HDFC", "RELIANCE"].
        weights:          Portfolio weights (fractions summing to 1.0),
                          e.g. [0.40, 0.30, 0.30].
        period:           Lookback for return calculation – e.g. '1y', '2y'.
        benchmark_ticker: Index ticker for beta calculation (default: ^NSEI = Nifty 50).

    Returns:
        JSON string with annualised volatility, portfolio beta, Sharpe ratio
        (approximate), HHI concentration score, and per-stock contribution.
    """
    try:
        holdings = [h.strip().upper() for h in holdings]
        weights = [float(w) for w in weights]

        # Validate
        if len(holdings) != len(weights):
            return json.dumps({"error": "holdings and weights must have the same length."})

        total_w = sum(weights)
        if abs(total_w - 1.0) > 0.05:
            # Auto-normalise
            weights = [w / total_w for w in weights]

        # ── Fetch daily close prices ───────────────────────────────────────────
        tickers_ns = [
            (h + ".NS" if "." not in h and not h.startswith("^") else h)
            for h in holdings
        ]
        all_tickers = tickers_ns + [benchmark_ticker]

        raw = yf.download(
            all_tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        # Handle single vs multi-ticker download format
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            prices = raw[["Close"]]
            prices.columns = [all_tickers[0]]

        prices = prices.dropna()

        if prices.empty or len(prices) < 30:
            return json.dumps({"error": "Not enough price history to compute risk metrics."})

        # ── Daily returns ──────────────────────────────────────────────────────
        returns = prices.pct_change().dropna()

        bench_col = benchmark_ticker
        stock_cols = tickers_ns

        # Filter to available columns
        available = [c for c in stock_cols if c in returns.columns]
        if not available:
            return json.dumps({"error": "Could not fetch return data for any of the provided tickers."})

        # Align weights to available tickers
        avail_idx = [tickers_ns.index(c) for c in available]
        avail_weights = [weights[i] for i in avail_idx]
        w_sum = sum(avail_weights)
        avail_weights = [w / w_sum for w in avail_weights]

        stock_returns = returns[available]

        # ── Portfolio returns ──────────────────────────────────────────────────
        port_returns = stock_returns.dot(avail_weights)

        # ── Annualised volatility ──────────────────────────────────────────────
        ann_vol = float(port_returns.std() * np.sqrt(252) * 100)  # in %

        # ── Per-stock volatility ───────────────────────────────────────────────
        per_stock_vol = (stock_returns.std() * np.sqrt(252) * 100).to_dict()

        # ── Portfolio Beta vs benchmark ────────────────────────────────────────
        portfolio_beta = None
        if bench_col in returns.columns:
            bench_returns = returns[bench_col]
            aligned = pd.concat([port_returns, bench_returns], axis=1).dropna()
            if len(aligned) > 20:
                cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
                bench_var = cov_matrix[1, 1]
                portfolio_beta = _safe_round(cov_matrix[0, 1] / bench_var if bench_var != 0 else None)

        # ── Approximate Sharpe (assume risk-free = 6.5% for India) ────────────
        rf_daily = 0.065 / 252
        excess = port_returns - rf_daily
        sharpe = _safe_round(
            float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() != 0 else None
        )

        # ── Herfindahl–Hirschman Index (concentration) ─────────────────────────
        hhi = _safe_round(sum(w ** 2 for w in avail_weights), 4)
        if hhi >= 0.25:
            concentration_label = "High concentration – consider diversification"
        elif hhi >= 0.15:
            concentration_label = "Moderate concentration"
        else:
            concentration_label = "Well-diversified"

        # ── Per-stock contribution to portfolio risk ───────────────────────────
        cov_matrix_stocks = stock_returns.cov() * 252
        w_arr = np.array(avail_weights)
        port_var = float(w_arr @ cov_matrix_stocks.values @ w_arr)
        marginal_contrib = (cov_matrix_stocks.values @ w_arr)
        risk_contrib = (w_arr * marginal_contrib / port_var * 100) if port_var > 0 else w_arr * 0

        per_stock = []
        for i, col in enumerate(available):
            per_stock.append({
                "ticker": col,
                "weight_pct": _safe_round(avail_weights[i] * 100),
                "annualised_vol_pct": _safe_round(per_stock_vol.get(col)),
                "risk_contribution_pct": _safe_round(float(risk_contrib[i])),
            })

        result = {
            "portfolio_summary": {
                "tickers": available,
                "weights_used": [_safe_round(w * 100) for w in avail_weights],
                "period": period,
                "as_of": datetime.utcnow().strftime("%Y-%m-%d"),
                "benchmark": benchmark_ticker,
            },
            "risk_metrics": {
                "annualised_volatility_pct": _safe_round(ann_vol),
                "portfolio_beta": portfolio_beta,
                "approx_sharpe_ratio": sharpe,
                "herfindahl_index": hhi,
                "concentration_label": concentration_label,
            },
            "per_stock_breakdown": per_stock,
        }
        return json.dumps(result, default=str)

    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ─────────────────────────────────────────────────────────────────────────────
# Entry point – run via stdio transport (used by langchain-mcp-adapters)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
