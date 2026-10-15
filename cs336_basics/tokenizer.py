import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(
        
):
    pass

def process_chunk(
    
):

    mini_chunk_size = 4096

    

    return NotImplementedError

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[
    dict[int,bytes],
    list[tuple[bytes, bytes]],
]:
    """Train Byte-pair Encoder, given corpus path & targeted vocab_size

    Args:
        - input_path : Path to corpus
        - vocab_size : Targetted output vocab size
        - special_tokens : List of special tokens

    Returns:
        - vocab: dict[int, bytes] : The resulting vocab after BPE merge
        - merges: list[tuple[bytes, bytes]] : List of BPE merges produces from training
    
    """

    num_workers = 8

    with open(input_path, "rb") as f:
        special_tokens_bytes = list(s.encode() for s in special_tokens)
        boundaries = find_chunk_boundaries(f, num_workers, "\u0003".encode());

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            process_chunk()
        
    raise NotImplementedError