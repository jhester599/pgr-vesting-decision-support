"""
Central configuration for the PGR Vesting Decision Support engine.

API keys are loaded from a .env file (never hardcoded).
All tuneable parameters for the backtest are defined here.

This package re-exports every name from the logical sub-modules so that
all existing ``import config`` / ``config.XXX`` call sites continue to work
without modification.
"""

from dotenv import load_dotenv

load_dotenv()

from .api import *       # noqa: F401, F403  — API credentials, URLs, EDGAR helpers, rate limits
from .features import *  # noqa: F401, F403  — FRED series, ETF universe, feature sets, paths
from .splits import *   # noqa: F401, F403  — canonical split registry, reviewed price jumps
from .model import *     # noqa: F401, F403  — WFO, ML diagnostics, calibration, conformal, BLP
from .tax import *       # noqa: F401, F403  — tax rates, RSU schedule, STCG guard, TLH
from .paths import *     # noqa: F401, F403  — production artifact paths (artifacts/)
