import sys
from pathlib import Path

from tqdm import tqdm
import torch
import pandas as pd

from lit_llama import Tokenizer

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


def prepare(destination_path: Path = Path("data/dnes")) -> None:
    destination_path.mkdir(parents=True, exist_ok=True)

    input_file_path = destination_path / "comment_list.csv"

    df = pd.read_csv(input_file_path, sep=",", encoding="utf-8")
    n = len(df)
    train_data = list(df[:int(n * 0.9)]["comment_text"])
    val_data = list(df[int(n * 0.9):]["comment_text"])

    tokenizer = Tokenizer(destination_path / "tokenizer.model")
    train_set = [tokenizer.encode(sample, eos=True, max_length=512)
                 for sample in tqdm(train_data)]
    torch.save(train_set, destination_path / "train.pt")
    val_set = [tokenizer.encode(sample, eos=True, max_length=512)
               for sample in tqdm(val_data)]
    torch.save(val_set, destination_path / "val.pt")


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
