#!/usr/bin/env bash
# exit on error
set -e

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create model directory if it doesn't exist
mkdir -p model

# Run the training script to generate model files
python train_model.py