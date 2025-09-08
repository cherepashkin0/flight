SELECT *
FROM `flight-cancellation-prediction.mlflow_tracking.mlflow_metrics_log`
WHERE project_name = 'flight_cancellation_project'
AND metric_name = 'f2_score'
ORDER BY metric_value DESC
LIMIT 5;
