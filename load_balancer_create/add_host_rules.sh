# Добавляем правило для API поддомена
gcloud compute url-maps add-host-rule flight-cancellation-url-map \
    --hosts=flight-cancellation-api.codezauber.de \
    --path-matcher-name=api-matcher

# Добавляем path matcher для API
gcloud compute url-maps add-path-matcher flight-cancellation-url-map \
    --path-matcher-name=api-matcher \
    --default-service=api-backend-service

# Добавляем правило для UI поддомена
gcloud compute url-maps add-host-rule flight-cancellation-url-map \
    --hosts=flight-cancellation-ui.codezauber.de \
    --path-matcher-name=ui-matcher

# Добавляем path matcher для UI
gcloud compute url-maps add-path-matcher flight-cancellation-url-map \
    --path-matcher-name=ui-matcher \
    --default-service=ui-backend-service

