#!/bin/bash

# Установите переменные
PROJECT_ID="flight-cancellation-prediction"
ZONE="europe-west3-a"
CLUSTER_NAME="flight-k8s"
NAMESPACE="prod"
PROJECT_NAME="flight"

echo "=== Подключение к кластеру ==="
gcloud container clusters get-credentials "$CLUSTER_NAME" \
  --zone "$ZONE" --project "$PROJECT_ID"

echo -e "\n=== Проверка состояния сервисов ==="
kubectl -n "$NAMESPACE" get svc

echo -e "\n=== Детали LoadBalancer сервисов ==="
kubectl -n "$NAMESPACE" get svc -o jsonpath='{range .items[?(@.spec.type=="LoadBalancer")]}{.metadata.name}{"\n  Status: "}{.status.loadBalancer}{"\n"}{end}'

echo -e "\n=== Проверка подов ==="
kubectl -n "$NAMESPACE" get pods -o wide

echo -e "\n=== Проверка endpoints ==="
kubectl -n "$NAMESPACE" get endpoints

echo -e "\n=== События в namespace ==="
kubectl -n "$NAMESPACE" get events --sort-by='.lastTimestamp' | tail -20

echo -e "\n=== Проверка квот GCP ==="
gcloud compute project-info describe --project=$PROJECT_ID | grep -A 5 "EXTERNAL"

echo -e "\n=== Проверка firewall правил ==="
gcloud compute firewall-rules list --project=$PROJECT_ID | grep -E "k8s|gke"

echo -e "\n=== Проверка forwarding rules ==="
gcloud compute forwarding-rules list --project=$PROJECT_ID

echo -e "\n=== Логи из GCP Load Balancer ==="
gcloud logging read "resource.type=gce_network_region" \
  --project=$PROJECT_ID \
  --limit=20 \
  --format=json | jq '.[] | {timestamp: .timestamp, message: .textPayload}'
