#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
echo "Downloading all CS:GO datasets..."
python download_all_datasets.py
