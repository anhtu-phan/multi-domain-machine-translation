#!/bin/bash

if [ -d "/home/students/anhtu/multi-domain-machine-translation/venv/" ]; then
  source /home/students/anhtu/multi-domain-machine-translation/venv/bin/activate
else
  python3 -m venv /home/students/anhtu/multi-domain-machine-translation/venv
  source /home/students/anhtu/multi-domain-machine-translation/venv/bin/activate
  pip install -r /home/students/anhtu/multi-domain-machine-translation/requirements.txt
fi
python /home/students/anhtu/multi-domain-machine-translation/eval.py --data_dir ./datasets/de-en/mixed --test_data_dir ./datasets/de-en/news --model_path ./checkpoints/model_de_en/model.pt
deactivate