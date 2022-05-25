#!/bin/bash

if [ -d "venv/" ]; then
  source venv/bin/activate
else
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
fi
python eval.py --data_dir ./datasets/de-en/mixed --test_data_dir ./datasets/de-en/news --model_path ./checkpoints/model_de_en/model.pt
deactivate