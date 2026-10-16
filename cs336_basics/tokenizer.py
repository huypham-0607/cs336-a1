import os
import multiprocessing
import re

from typing import BinaryIO


class Tokenizer():

    PAT = PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    BASE_VOCAB_SIZE = 256

    def find_chunk_boundaries(
        self,
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

    def pretokenize_chunk(
        self,
        file: BinaryIO,
        start: int,
        end: int,
        keep_special: bool = True,
    ):
        """Process a chunk of data in file, outlined by [start, end)

        This function strips 

        """

        # Splitting on special character
        if keep_special:
            split_pat = re.compile("(" + "|".join(re.escape(special_token) for special_token in self.special_tokens) + ")")
        else:
            split_pat = re.compile("|".join(re.escape(special_token) for special_token in self.special_tokens))

        corpus = file.read(end-start).decode('utf-8', errors='ignore')
        words = list(re.split(split_pat, corpus))

        # Pretokenization
        tokens = []
        for word in words:
            tokens.append(word if word in self.special_tokens else re.findall(Tokenizer.PAT, word))

        return tokens


    def map_word(
        self,        
        word: str,
    ) -> list[int]:
        """Map each bytes in a word to its corresponding value in the vocab 
        """

        if word in self.special_tokens:
            return [Tokenizer.BASE_VOCAB_SIZE + self.special_tokens.index(word)]

        return list(int(byte) for byte in word.encode())


    def encode():
        """Remember: split the chunk by special char, where special char is its own element.
        """
        raise NotImplementedError

    def get_freq(
        self,
        mapped_words: list[list[int]],
    ) -> dict[tuple[int,int], int]:
        """Get frequency count for each adjacent token pairs
        """

        freq = dict()

        for word in mapped_words:
            for pair in zip(word[:-1], word[1:]):
                freq[pair] += 1
        
        return freq

    def merge_pair(
        self,
        mapped_words: list[list[int]],
        pair: tuple[int,int],
        replace_id: int,
    ) -> None:
        """Modify mapped_words in-place, merging every occurences of pair
        """
        for idx in range(len(mapped_words)):
            new_word = []
            for x in mapped_words[idx]:
                if len(new_word) and new_word[-1] == pair[0] and x == pair[1]:
                    new_word[-1] = replace_id
                    continue
                new_word.append(x)
            mapped_words[idx] = new_word

    def train_bpe(
        self,
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

        self.special_tokens = special_tokens
        self.vocab_size = vocab_size
        num_workers = 8

        words = []

        # Getting chunk boundaries for parallelism
        with open(input_path, "rb") as f:
            special_tokens_bytes = list(s.encode() for s in special_tokens)
            boundaries = self.find_chunk_boundaries(f, num_workers, "\u0003".encode());

        # Adding special tokens to vocab
        self.vocab = dict((i,bytes(i)) for i in range(Tokenizer.BASE_VOCAB_SIZE))
        for i in range(len(special_tokens)):
            self.vocab[Tokenizer.BASE_VOCAB_SIZE + i] = special_tokens[i].encode()

        mapped_words = []
        self.merges = []

        # Pretokenize chunks & map pretokenized words to the vocab
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            with open(input_path, "rb") as f:
                words = self.pretokenize_chunk(f, start, end, False)
                mapped_words.append(self.map_word(word) for word in words)

        # Start merging adjacent pairs
        while (len(self.vocab) <= vocab_size):
            freq = self.get_freq(mapped_words)
            max_pair = max(freq, key=lambda k: (freq[k],k))

            key = len(self.vocab)
            value = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.vocab[key] = value

            self.merges.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))

            self.merge_pair(mapped_words, max_pair, key)

        return self.vocab, self.merges