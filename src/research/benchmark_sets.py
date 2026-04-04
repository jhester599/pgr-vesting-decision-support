"""Shared benchmark-family definitions for v9 research scripts."""

from __future__ import annotations


BENCHMARK_FAMILIES: dict[str, str] = {
    "VTI": "broad_equity",
    "VOO": "broad_equity",
    "VXUS": "broad_equity",
    "VEA": "broad_equity",
    "VWO": "broad_equity",
    "VGT": "sector",
    "VHT": "sector",
    "VFH": "sector",
    "VIS": "sector",
    "VDE": "sector",
    "VPU": "sector",
    "KIE": "sector",
    "VIG": "dividend",
    "SCHD": "dividend",
    "BND": "fixed_income",
    "BNDX": "fixed_income",
    "VCIT": "fixed_income",
    "VMBS": "fixed_income",
    "VNQ": "real_asset",
    "GLD": "real_asset",
    "DBC": "real_asset",
}


BENCHMARK_POOLS: dict[str, list[str]] = {
    "broad_equity": ["VTI", "VOO", "VXUS", "VEA", "VWO"],
    "sector": ["VGT", "VHT", "VFH", "VIS", "VDE", "VPU", "KIE"],
    "dividend": ["VIG", "SCHD"],
    "fixed_income": ["BND", "BNDX", "VCIT", "VMBS"],
    "real_asset": ["VNQ", "GLD", "DBC"],
    "risk_assets_core": [
        "VTI",
        "VOO",
        "VXUS",
        "VEA",
        "VWO",
        "VGT",
        "VHT",
        "VFH",
        "VIS",
        "VDE",
        "VPU",
        "KIE",
        "VIG",
        "SCHD",
    ],
    "defensive_assets": ["BND", "BNDX", "VCIT", "VMBS", "VNQ", "GLD", "DBC"],
}
