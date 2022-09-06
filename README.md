# multi-domain-machine-translation

## How to run

- The trained models can be downloaded from  <a href='https://drive.google.com/file/d/1CA0FV9Dnu0xsL6v5OurE_KIOTOR2z_tE/view?usp=sharing'>here</a>

### Install 

    #python3.7
    pip install --upgrade pip
    pip install -r requirements.txt

### Demo
    
    python run_demo_server.py --port PORT --data_dir_mixed PATH --data_dir_domain PATH

- `PORT`: port to run server (default server will run on http://localhost:9595)
- `data_dir_mixed`: folder store mixed dataset to run original transformer architecture
- `data_dir_domain`: list folder store domain data

Example: Download trained models from <a href='https://drive.google.com/file/d/1CA0FV9Dnu0xsL6v5OurE_KIOTOR2z_tE/view?usp=sharing'>here</a> and place to project folder and run command:
`python run_demo_server.py --data_dir_mixed ./datasets/de-en/mixed --data_dir_domain ./datasets/de-en/news ./datasets/de-en/ted` 

### Training
    
    python training.py --data_dir PATH --model_type MODEL_TYPE

- `data_dir`:
  - mixed dataset if training with original transformer architecture
  - list of domain data if training with domain proportion
- `model_type`:
  + `0`: direct-training (original transformer)
  + `1`: edc (modified transformer with domain proportion plugged in both encoder and decoder)
  + `2`: encoder (modified transformer with domain proportion plugged just in encoder)

`python training.py --data_dir ./datasets/de-en/mixed --model_type 0`: training with original transformer

`python training.py --data_dir ./datasets/de-en/news ./datasets/de-en/ted --model_type 1`: training with edc

`python training.py --data_dir ./datasets/de-en/news ./datasets/de-en/ted --model_type 2`: training with encoder



### Eval
    
    python evaluate.py --data_dir PATH --test_data_dir PATH --model_path PATH --model_type MODEL_TYPE
- `data_dir`: data folder using when training
  - mixed dataset if training with original transformer architecture
  - list of domain data if training with domain proportion
- `test_data_dir`: data folder contain test set
- `model_path`: path of model checkpoint
- `model_type`:
  + `0`: direct-training (original transformer)
  + `1`: edc (modified transformer with domain proportion plugged in both encoder and decoder)
  + `2`: encoder (modified transformer with domain proportion plugged just in encoder)

`python eval.py --data_dir ./datasets/de-en/news ./datasets/de-en/ted --test_data_dir ./datasets/de-en/ted --model_path ./checkpoints/model_de_en/model_mutil.pt --model_type 1`: evaluate model in ted domain
