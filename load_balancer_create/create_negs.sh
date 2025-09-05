# NEG для API сервиса
gcloud compute network-endpoint-groups create api-neg \
    --region=europe-west3 \
    --network-endpoint-type=serverless \
    --cloud-run-service=flight-api

# NEG для UI сервиса
gcloud compute network-endpoint-groups create ui-neg \
    --region=europe-west3 \
    --network-endpoint-type=serverless \
    --cloud-run-service=flight-ui
