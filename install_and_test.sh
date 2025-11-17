#!/bin/bash

echo "============================================================"
echo "Installing dependencies and running test..."
echo "============================================================"
echo

echo "[Step 1] Upgrading tokenizers and transformers..."
pip install --upgrade tokenizers transformers --quiet

echo
echo "[Step 2] Installing other dependencies..."
pip install -r requirements.txt --quiet

echo
echo "[Step 3] Installing package..."
pip install -e . --quiet

echo
echo "[Step 4] Running test..."
echo
python test.py

