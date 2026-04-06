"""
Tax rates, RSU vesting schedule, STCG boundary guard, and TLH parameters.
"""

import os

# ---------------------------------------------------------------------------
# Tax rates (federal maximums; add state rate in .env as needed)
# ---------------------------------------------------------------------------
LTCG_RATE: float = float(os.getenv("LTCG_RATE", "0.20"))
STCG_RATE: float = float(os.getenv("STCG_RATE", "0.37"))

# ---------------------------------------------------------------------------
# RSU vesting schedule
# ---------------------------------------------------------------------------
TIME_RSU_VEST_MONTH: int = 1    # January (time-based)
TIME_RSU_VEST_DAY: int = 19
PERF_RSU_VEST_MONTH: int = 7    # July (performance-based)
PERF_RSU_VEST_DAY: int = 17

# ---------------------------------------------------------------------------
# v4.4 — STCG Tax Boundary Guard
# ---------------------------------------------------------------------------
# Minimum predicted 6M alpha required to justify selling a lot still in the
# STCG zone (held 6–12 months) rather than waiting for LTCG qualification.
#
# Rationale: selling STCG vs. LTCG costs ~17–22pp in effective tax rate for
# most high-income earners (37% ordinary − 20% LTCG = 17pp; add 3.8% NIIT
# and state taxes for an upper bound near 22pp).  0.18 is the mid-range
# breakeven: if the model predicts less than 18% alpha, the tax savings from
# waiting a few weeks/months to cross the 365-day threshold likely exceed the
# opportunity cost of holding the concentrated position slightly longer.
#
# The 6–12 month zone is defined as: 180 < holding_days_at_vest <= 365.
# Lots held < 180 days have too long to wait; lots > 365 days are LTCG.
STCG_BREAKEVEN_THRESHOLD: float = 0.18
# Lower bound of the STCG boundary zone (days held, exclusive).
STCG_ZONE_MIN_DAYS: int = 180
# Upper bound of the STCG boundary zone — day 365 triggers LTCG.
STCG_ZONE_MAX_DAYS: int = 365

# ---------------------------------------------------------------------------
# v4.0 Tax-Loss Harvesting parameters
# ---------------------------------------------------------------------------
TLH_LOSS_THRESHOLD: float = -0.10       # Harvest when unrealized return < -10%
TLH_WASH_SALE_DAYS: int = 31            # Minimum days before repurchasing original
