# Добавляем API NEG к API backend
gcloud compute backend-services add-backend api-backend-service \
    --global \
    --network-endpoint-group=api-neg \
    --network-endpoint-group-region=europe-west3

# Добавляем UI NEG к UI backend
gcloud compute backend-services add-backend ui-backend-service \
    --global \
    --network-endpoint-group=ui-neg \
    --network-endpoint-group-region=europe-west3
