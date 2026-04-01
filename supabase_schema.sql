-- ============================================================
--  Power Anomaly Detection — Supabase Schema
--  Run this ONCE in: Supabase → SQL Editor → New Query → Run
-- ============================================================

-- 1. Create the predictions table ─────────────────────────────
CREATE TABLE IF NOT EXISTS public.predictions (
  id                     BIGSERIAL    PRIMARY KEY,
  timestamp              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
  global_active_power    FLOAT8       NOT NULL,
  global_reactive_power  FLOAT8       NOT NULL,
  voltage                FLOAT8       NOT NULL,
  global_intensity       FLOAT8       NOT NULL,
  sub_metering_1         FLOAT8       NOT NULL,
  sub_metering_2         FLOAT8       NOT NULL,
  sub_metering_3         FLOAT8       NOT NULL,
  anomaly_score          FLOAT8       NOT NULL,
  is_anomaly             BOOLEAN      NOT NULL
);

-- 2. Enable Row Level Security ─────────────────────────────────
ALTER TABLE public.predictions ENABLE ROW LEVEL SECURITY;

-- 3. RLS Policies ──────────────────────────────────────────────
-- Allow anon + authenticated to INSERT (backend uses service key,
-- but anon key must also be permitted for direct client inserts)
CREATE POLICY "Allow anon insert"
  ON public.predictions
  FOR INSERT
  TO anon, authenticated
  WITH CHECK (true);

-- Allow anon + authenticated to SELECT (frontend reads via anon key)
CREATE POLICY "Allow anon select"
  ON public.predictions
  FOR SELECT
  TO anon, authenticated
  USING (true);

-- 4. Enable Realtime ───────────────────────────────────────────
-- Allows the frontend Supabase client to subscribe to live inserts.
ALTER PUBLICATION supabase_realtime ADD TABLE public.predictions;

-- 5. Index on timestamp for fast ORDER BY queries ──────────────
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
  ON public.predictions (timestamp DESC);

-- 6. Verify ────────────────────────────────────────────────────
-- After running, you should see the table in:
--   Table Editor → predictions (9 data columns + id + timestamp)
--   Authentication → Policies → predictions (2 policies)
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public' AND table_name = 'predictions'
ORDER BY ordinal_position;
