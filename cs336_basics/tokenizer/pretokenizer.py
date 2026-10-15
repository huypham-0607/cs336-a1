

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[
    dict[int,bytes],
    list[tuple[bytes, bytes]],
]:
    raise NotImplementedError