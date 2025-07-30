import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("olimpic_dir", type=str, help="Root directory path of olimpic-icdar24 repo")
parser.add_argument("vocab_path", type=str, help="Path to save vocab file")
args = parser.parse_args()
sys.path.append(args.olimpic_dir)

from app.linearization.vocabulary import print_vocabulary

with open(args.vocab_path, "w") as f:
    f.write("<bos>\n")
    f.write("<pad>\n")
    print_vocabulary(file=f)
    f.write("<eos>\n")