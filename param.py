import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", help = 'either lstm_lstm or lstm_lstm_attn or bert_frozen or bert_tuned')
    parser.add_argument("--bert_model_type", help = 'either frozen or tuned')
    parser.add_argument("--model_op_path")
    parser.add_argument("--preprocessed_dir")
    args = parser.parse_args()

    return args

args = parse_args()



