#!/bin/bash

PROJECT_ID="flight-cancellation-prediction"
CLUSTER_NAME="flight-k8s"
ZONE="europe-west3-a"

echo "=== Проверяем, какой Service Account используют узлы GKE ==="

# Получаем информацию о node pool
NODE_SA=$(gcloud container node-pools describe default-pool \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --format="value(config.serviceAccount)")

if [ -z "$NODE_SA" ] || [ "$NODE_SA" == "default" ]; then
    # Используется Compute Engine default service account
    NODE_SA=$(gcloud compute project-info describe \
        --project=$PROJECT_ID \
        --format="value(defaultServiceAccount)")
fi

echo "Service Account узлов: $NODE_SA"

echo -e "\n=== Проверяем права этого SA ==="
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --format="table(bindings.role)" \
    --filter="bindings.members:serviceAccount:${NODE_SA}" | head -20

echo -e "\n=== Добавляем необходимые права для BigQuery ==="
echo "Добавляем роли для $NODE_SA..."

# BigQuery Data Viewer
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${NODE_SA}" \
    --role="roles/bigquery.dataViewer" \
    --condition=None

# BigQuery Job User (для выполнения запросов)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${NODE_SA}" \
    --role="roles/bigquery.jobUser" \
    --condition=None

# Storage Object Viewer (для доступа к GCS)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${NODE_SA}" \
    --role="roles/storage.objectViewer" \
    --condition=None

echo -e "\n=== Перезапускаем поды для применения изменений ==="
kubectl rollout restart deployment/api-flight -n prod

echo -e "\n=== Ждем перезапуска ==="
kubectl rollout status deployment/api-flight -n prod --timeout=300s

echo -e "\n=== Проверяем статус ==="
kubectl get pods -n prod -l app=api-flight

echo -e "\n⚠️  ВАЖНО: Этот метод даст доступ ВСЕМ подам в кластере!"
echo "Для production используйте Workload Identity или отдельные Service Accounts"