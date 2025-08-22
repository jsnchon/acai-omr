import argparse
from olimpic_app.linearization.vocabulary import print_vocabulary

parser = argparse.ArgumentParser()
parser.add_argument("vocab_path", type=str, help="Path to save vocab file")
args = parser.parse_args()

with open(args.vocab_path, "w") as f:
    f.write("<bos>\n")
    f.write("<pad>\n")
    f.write("<eos>\n")
    print_vocabulary(file=f)