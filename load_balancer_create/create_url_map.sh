# Создаем URL map с UI как default service
gcloud compute url-maps create flight-cancellation-url-map \
    --default-service=ui-backend-service
