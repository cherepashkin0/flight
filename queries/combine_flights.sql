CREATE OR REPLACE TABLE `flight-cancellation-prediction.flight_data.combined_flights` AS
SELECT * FROM `flight-cancellation-prediction.flight_data.combined_flights_2018`
UNION ALL
SELECT * FROM `flight-cancellation-prediction.flight_data.combined_flights_2019`
UNION ALL
SELECT * FROM `flight-cancellation-prediction.flight_data.combined_flights_2020`
UNION ALL
SELECT * FROM `flight-cancellation-prediction.flight_data.combined_flights_2021`
UNION ALL
SELECT * FROM `flight-cancellation-prediction.flight_data.combined_flights_2022`
