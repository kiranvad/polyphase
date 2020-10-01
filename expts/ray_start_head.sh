#!/bin/bash

echo "starting ray head node"
# Launch the head node
ray start --head --webui-host 0.0.0.0 --node-ip-address=$1 --port=6379 --redis-password=$2
sleep infinity