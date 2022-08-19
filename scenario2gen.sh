#!/usr/bin/env bash
set -euo pipefail

# scenario 2
pipenv run python main.py \
  --fed gen \
  --scale false \
  --cache true \
  --batch_size 96 \
  --rounds 20 \
  --timeit true \
  --epochs 1 \
  --test_sets Lithuania  \
  --fed_clients Serbia Austria Switzerland \
  --fedgen_config "config/fedgen_config.yaml" \
  --backbone resnet50 \
  --device cuda \
  --workers 6 \
  --data_root "/home/ubuntu/BigEarthNet/datasets/" \
  --output_dir "/home/ubuntu/BigEarthNet/experiments/clients/" \
  --fed_servers_output_dir "/home/ubuntu/BigEarthNet/experiments/servers/" \
  --fed_clients_source_dir "/home/ubuntu/BigEarthNet/experiments/clients/"

# Finland Portugal Ireland Lithuania Serbia Austria Switzerland

# Lambda:
#  --data_root "/home/ubuntu/BigEarthNet/datasets/" \
#  --output_dir "/home/ubuntu/BigEarthNet/experiments/clients/" \
#  --fed_servers_output_dir "/home/ubuntu/BigEarthNet/experiments/servers/" \
#  --fed_clients_source_dir "/home/ubuntu/BigEarthNet/experiments/clients/"

# Local machine:
#  --data_root "/home/amer/data/BigEarthNet/nvme/datasets/" \
#  --output_dir "/home/amer/data/BigEarthNet/nvme/experiments/clients/" \
#  --fed_servers_output_dir "/home/amer/data/BigEarthNet/nvme/experiments/servers/" \
#  --fed_clients_source_dir "/home/amer/data/BigEarthNet/nvme/experiments/clients/"
