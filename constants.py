MODEL_TYPE = ['direct', 'edc', 'encoder']
CONFIG = {
    "LEARNING_RATE": 1e-7,
    "BATCH_SIZE": 32,
    "HID_DIM": 512,
    "ENC_LAYERS": 6,
    "DEC_LAYERS": 6,
    "ENC_HEADS": 4,
    "DEC_HEADS": 4,
    "ENC_PF_DIM": 1024,
    "DEC_PF_DIM": 1024,
    "ENC_DROPOUT": 0.2,
    "DEC_DROPOUT": 0.2,
    "N_EPOCHS": 1000000,
    "DOMAIN_EPS": 0.05,
    "CLIP": 1,
    "MODEL_TYPE": MODEL_TYPE[1]
}
