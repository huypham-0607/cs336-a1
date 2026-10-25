import pytest
from cs336_basics import tokenizer 

TEXT_FILE = "cs336_basics/smoke_test_corpus.txt"

def test_run():
    vocab, merges = tokenizer.train_bpe(TEXT_FILE, 261, ["<|endoftext|>"])

    print(vocab)
    print(merges)
