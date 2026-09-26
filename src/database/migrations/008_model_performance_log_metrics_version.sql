-- Review 2026-09-25, step 5 (WP7): tag each model-health row with the metric
-- definitions it was computed under.
--
-- Rows written before this migration used the old definitions: an OOS-R^2
-- naive benchmark that contained the target it was scored against (F04),
-- ensemble weights and shrinkage chosen on the whole OOS history, ECE measured
-- on the calibrator's own training rows, and conformal intervals calibrated
-- on residuals not yet realised (F13). They are kept as the record of what
-- those runs reported and tagged 'pre-2026-09-25'. New rows carry
-- config.MODEL_HEALTH_METRICS_VERSION; the drift monitor compares only rows
-- that share a version.
ALTER TABLE model_performance_log ADD COLUMN metrics_version TEXT;

UPDATE model_performance_log
SET metrics_version = 'pre-2026-09-25'
WHERE metrics_version IS NULL;
