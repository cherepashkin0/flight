PROJECT_ID="flight-cancellation-pred"
SA_RUN="flight-api-sa@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_RUN}" --role="roles/artifactregistry.reader"
