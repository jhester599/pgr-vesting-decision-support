-- Migration 003: model_retrain_log table (v35.1 — Tier 5.4 retrain trigger)
CREATE TABLE IF NOT EXISTS model_retrain_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    triggered_at    TEXT NOT NULL,
    breach_streak   INTEGER NOT NULL,
    triggered       INTEGER NOT NULL,
    cooldown_active INTEGER NOT NULL,
    last_trigger_date TEXT,
    notes           TEXT,
    created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
