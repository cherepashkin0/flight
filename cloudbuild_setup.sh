# 1) Set vars
PROJECT_ID="flight-cancellation-pred"
REGION="europe-west4"
REPO="flight"
SA_RUN_NAME="flight-api-sa"
SA_RUN="${SA_RUN_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"

# 2) Get the correct Cloud Build SA (note: it's PROJECT_NUMBER@cloudbuild.gserviceaccount.com)
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SA_BUILD="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

# 3) Make sure a first build has run (this auto-creates the CB SA)
if ! gcloud iam service-accounts describe "$SA_BUILD" >/dev/null 2>&1; then
  echo "Cloud Build SA not found yet; running a tiny one-off build to create it..."
  tmpdir="$(mktemp -d)"
  cat > "${tmpdir}/cloudbuild.yaml" <<'YAML'
steps:
- name: gcr.io/google.com/cloudsdktool/cloud-sdk
  args: ["gcloud","version"]
YAML
  gcloud builds submit "$tmpdir" --config "${tmpdir}/cloudbuild.yaml"
fi

# 4) Create runtime SA (idempotent)
gcloud iam service-accounts create "$SA_RUN_NAME" \
  --display-name="Cloud Run runtime SA" || true

# 5) Grant runtime SA the app's needed reads (BQ + GCS)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_RUN}" --role="roles/bigquery.jobUser"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_RUN}" --role="roles/bigquery.dataViewer"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_RUN}" --role="roles/storage.objectViewer"

# 6) Grant Cloud Build SA deploy/push rights
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_BUILD}" --role="roles/run.admin"
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_BUILD}" --role="roles/artifactregistry.writer"

# 7) Allow Cloud Build SA to impersonate the runtime SA at deploy
gcloud iam service-accounts add-iam-policy-binding "$SA_RUN" \
  --member="serviceAccount:${SA_BUILD}" --role="roles/iam.serviceAccountUser"
