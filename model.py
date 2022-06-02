from transformer_pytorch.transformer import Encoder, Decoder, Seq2Seq
from transformer_pytorch.domain_mixing_transformer import Encoder as DomainEncoder, Decoder as DomainDecoder, Seq2Seq as DomainSeq2Seq
from constants import MODEL_TYPE


def load_model(input_dim, output_dim, src_pad_idx, trg_pad_idx, config, nb_domain, device):
    if nb_domain > 1:
        enc = DomainEncoder(input_dim, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'],
                            config['ENC_PF_DIM'], config['ENC_DROPOUT'], nb_domain, config['DOMAIN_EPS'], device)
        if config['MODEL_TYPE'] == MODEL_TYPE[2]:
            print(f"{'-' * 10}Construct domain mixing ENCODER network{'-' * 10}")
            dec = Decoder(output_dim, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                          config['DEC_PF_DIM'], config['DEC_DROPOUT'], device)
            model = DomainSeq2Seq(enc, dec, src_pad_idx, trg_pad_idx, True, device).to(device)
        else:
            print(f"{'-' * 10}Construct domain mixing EDC network{'-' * 10}")
            dec = DomainDecoder(output_dim, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'],
                                config['DEC_PF_DIM'], config['DEC_DROPOUT'], nb_domain, config['DOMAIN_EPS'], device)
            model = DomainSeq2Seq(enc, dec, src_pad_idx, trg_pad_idx, False, device).to(device)
    else:
        print(f"{'-' * 10}Construct original network{'-' * 10}")
        enc = Encoder(input_dim, config['HID_DIM'], config['ENC_LAYERS'], config['ENC_HEADS'], config['ENC_PF_DIM'],
                      config['ENC_DROPOUT'], device)
        dec = Decoder(output_dim, config['HID_DIM'], config['DEC_LAYERS'], config['DEC_HEADS'], config['DEC_PF_DIM'],
                      config['DEC_DROPOUT'], device)
        model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_idx, device).to(device)

    return model
