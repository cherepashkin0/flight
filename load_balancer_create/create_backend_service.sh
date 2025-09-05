# Backend service для API
gcloud compute backend-services create api-backend-service \
    --global \
    --load-balancing-scheme=EXTERNAL_MANAGED

# Backend service для UI
gcloud compute backend-services create ui-backend-service \
    --global \
    --load-balancing-scheme=EXTERNAL_MANAGED
