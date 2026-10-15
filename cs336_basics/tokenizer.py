import os
import multiprocessing
import re

from typing import BinaryIO


class Tokenizer():

    PAT = PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    @staticmethod
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

    @staticmethod
    def pretokenize_chunk(
        file: BinaryIO,
        start: int,
        end: int,
        special_tokens: list[str]
    ):
        """Process a chunk of data in file, outlined by [start, end)

        This function strips 

        """

        # Splitting on special character
        split_pat = re.compile("|".join(re.escape(special_token) for special_token in special_tokens))

        corpus = file.read(end-start).decode('utf-8', errors='ignore')
        words = list(re.split(split_pat, corpus))

        # Pretokenization
        tokens = []
        for word in words:
            tokens.append(re.findall(Tokenizer.PAT, word))

        return tokens

    @staticmethod
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

        words = []

        with open(input_path, "rb") as f:
            special_tokens_bytes = list(s.encode() for s in special_tokens)
            boundaries = Tokenizer.find_chunk_boundaries(f, num_workers, "\u0003".encode());

        freq = dict()

        vocab = dict((i,bytes(i)) for i in range(256))

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            with open(input_path, "rb") as f:
                words_bytes = (s.encode() for s in Tokenizer.pretokenize_chunk(f, start, end, special_tokens))

                for wb in words_bytes:
                    words.append(list(vocab[int(b)] for b in words_bytes))
                


        merges = []

        for word in 
        
        raise NotImplementedError