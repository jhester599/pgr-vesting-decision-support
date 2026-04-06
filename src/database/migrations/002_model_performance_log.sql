CREATE TABLE IF NOT EXISTS model_performance_log (
    month_end                            TEXT PRIMARY KEY,
    aggregate_oos_r2                     REAL,
    aggregate_nw_ic                      REAL,
    aggregate_hit_rate                   REAL,
    ece                                  REAL,
    ece_ci_lower                         REAL,
    ece_ci_upper                         REAL,
    conformal_target_coverage            REAL,
    conformal_empirical_coverage         REAL,
    conformal_trailing_empirical_coverage REAL,
    conformal_trailing_coverage_gap      REAL,
    created_at                           TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
