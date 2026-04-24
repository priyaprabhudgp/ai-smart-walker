#!/bin/bash
sleep 10
SSID=$(iwgetid -r)

if [ "$SSID" = "CUHSD-Guest" ]; then
    echo "School network detected, accepting captive portal..."
    curl -s -X POST "http://PLACEHOLDER_URL" \
         --data "accept=true" \
         -L \
         --max-time 10
fi
