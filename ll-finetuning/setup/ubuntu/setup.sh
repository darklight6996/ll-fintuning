#!/bin/bash
echo "====================================================="
echo "  Ubuntu / Linux Setup Script for ll-finetuning"
echo "====================================================="

echo "[*] Installing Python dependencies..."
pip install -r requirements.txt

echo "[*] Running Assessment Tools Pre-Flight Check..."
python3 ../check_tools.py

echo "Setup complete."
