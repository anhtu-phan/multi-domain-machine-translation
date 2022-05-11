#!/bin/bash

if [ -d "/home/students/anhtu/multi-domain-machine-translation/venv/" ]; then
  source /home/students/anhtu/multi-domain-machine-translation/venv/bin/activate
else
  python3 -m venv /home/students/anhtu/multi-domain-machine-translation/venv
  source /home/students/anhtu/multi-domain-machine-translation/venv/bin/activate
  pip install -r /home/students/anhtu/multi-domain-machine-translation/requirements.txt
  python -m spacy download en_core_web_sm
  python -m spacy download de_core_news_sm
fi
python /home/students/anhtu/multi-domain-machine-translation/training.py --data_dir ./datasets/de-en/news ./datasets/de-en/ted
deactivate