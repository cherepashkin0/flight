CREATE OR REPLACE FUNCTION `flight_data.time_to_minutes`(time_val FLOAT64)
RETURNS INT64
AS (
  (
    WITH time_int AS (
      SELECT CAST(time_val AS INT64) AS t
    )
    SELECT
      CASE
        WHEN time_val IS NULL THEN NULL
        WHEN t = 2400 THEN 0
        WHEN t < 0 OR t > 2400 THEN NULL
        WHEN MOD(t, 100) >= 60 THEN NULL  -- e.g., 2365 is invalid (65 minutes)
        ELSE CAST(FLOOR(t / 100) AS INT64) * 60 + CAST(MOD(t, 100) AS INT64)
      END
    FROM time_int
  )
);

SELECT
  CRSDepTime,
  CAST(`flight_data.time_to_minutes`(CRSDepTime) AS INT64) AS CRSDepTime_minutes,

  DepTime,
  CAST(`flight_data.time_to_minutes`(DepTime) AS INT64) AS DepTime_minutes,

  WheelsOff,
  CAST(`flight_data.time_to_minutes`(WheelsOff) AS INT64) AS WheelsOff_minutes,

  WheelsOn,
  CAST(`flight_data.time_to_minutes`(WheelsOn) AS INT64) AS WheelsOn_minutes,

  CRSArrTime,
  CAST(`flight_data.time_to_minutes`(CRSArrTime) AS INT64) AS CRSArrTime_minutes,

  ArrTime,
  CAST(`flight_data.time_to_minutes`(ArrTime) AS INT64) AS ArrTime_minutes
FROM
  `flight-cancellation-prediction.flight_data.flights_all`;

-- Part 2: Convert time block ranges to start, end, and elapsed (as INT64)
SELECT
  DepTimeBlk,
  CAST(`flight_data.time_to_minutes`(CAST(SPLIT(DepTimeBlk, '-')[OFFSET(0)] AS FLOAT64)) AS INT64) AS DepTimeBlk_minutes_start,
  CAST(`flight_data.time_to_minutes`(CAST(SPLIT(DepTimeBlk, '-')[OFFSET(1)] AS FLOAT64)) AS INT64) AS DepTimeBlk_minutes_finish,
  CAST(
    `flight_data.time_to_minutes`(CAST(SPLIT(DepTimeBlk, '-')[OFFSET(1)] AS FLOAT64)) -
    `flight_data.time_to_minutes`(CAST(SPLIT(DepTimeBlk, '-')[OFFSET(0)] AS FLOAT64))
    AS INT64
  ) AS DepTimeBlk_minutes_elapsed,

  ArrTimeBlk,
  CAST(`flight_data.time_to_minutes`(CAST(SPLIT(ArrTimeBlk, '-')[OFFSET(0)] AS FLOAT64)) AS INT64) AS ArrTimeBlk_minutes_start,
  CAST(`flight_data.time_to_minutes`(CAST(SPLIT(ArrTimeBlk, '-')[OFFSET(1)] AS FLOAT64)) AS INT64) AS ArrTimeBlk_minutes_finish,
  CAST(
    `flight_data.time_to_minutes`(CAST(SPLIT(ArrTimeBlk, '-')[OFFSET(1)] AS FLOAT64)) -
    `flight_data.time_to_minutes`(CAST(SPLIT(ArrTimeBlk, '-')[OFFSET(0)] AS FLOAT64))
    AS INT64
  ) AS ArrTimeBlk_minutes_elapsed
FROM
  `flight-cancellation-prediction.flight_data.flights_all`;
