#!/bin/bash

NAMESPACE="prod"
PROJECT_NAME="flight"

echo "=== Проверка логов падающего API пода ==="
POD_NAME=$(kubectl -n $NAMESPACE get pods -l app=api-$PROJECT_NAME -o jsonpath='{.items[0].metadata.name}')
echo "Checking pod: $POD_NAME"

echo -e "\n=== Последние логи контейнера ==="
kubectl -n $NAMESPACE logs $POD_NAME --tail=50

echo -e "\n=== Предыдущие логи (если под перезапускался) ==="
kubectl -n $NAMESPACE logs $POD_NAME --previous --tail=50 2>/dev/null || echo "No previous logs available"

echo -e "\n=== Описание пода ==="
kubectl -n $NAMESPACE describe pod $POD_NAME | grep -A 10 "Events:"

echo -e "\n=== Проверка образа Docker локально (если возможно) ==="
echo "Попробуйте запустить контейнер локально:"
echo "docker run -it --rm -p 8001:8001 us-central1-docker.pkg.dev/flight-cancellation-prediction/flight/my_app:0.1"

echo -e "\n=== Проверка ресурсов ==="
kubectl -n $NAMESPACE top pod $POD_NAME 2>/dev/null || echo "Metrics not available"

echo -e "\n=== Проверка переменных окружения ==="
kubectl -n $NAMESPACE get pod $POD_NAME -o jsonpath='{.spec.containers[0].env}' | jq '.'