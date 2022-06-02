#!/bin/bash

if [ -d "venv/" ]; then
  source venv/bin/activate
else
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
fi
python eval.py --data_dir ./datasets/de-en/news ./datasets/de-en/ted --test_data_dir ./datasets/de-en/ted --model_path ./checkpoints/model_de_en/model_mutil.pt --model_type 1
deactivate