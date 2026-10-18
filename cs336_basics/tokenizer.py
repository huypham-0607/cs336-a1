import os
import multiprocessing
import regex as re

from queue import PriorityQueue
from multiprocessing import Pool, Process
from typing import BinaryIO

class TokenizerTrainer():

    # === Constants ====================================================================================

    PAT = PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    BASE_VOCAB_SIZE = 256

    # === BPE Trainer Helper functions ==================================================================

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
        file_path: str,
        start: int,
        end: int,
        keep_special: bool = True,
    ) -> list[str]:
        """Process a chunk of data in file, outlined by [start, end)

        This function either strips all special tokens, or keep them as a monolithic element

        """
        # Splitting on special character
        with open(file_path, "rb") as file:
            if keep_special:
                split_pat = re.compile("(" + "|".join(re.escape(special_token) for special_token in self.special_tokens) + ")")
            else:
                split_pat = re.compile("|".join(re.escape(special_token) for special_token in self.special_tokens))

            file.seek(start, os.SEEK_SET)
            corpus = file.read(end-start).decode('utf-8', errors='ignore')
            words = list(re.split(split_pat, corpus))

            # Pretokenization
            tokens = []
            for word in words:
                tokens += (word if word in self.special_tokens else re.findall(TokenizerTrainer.PAT, word))

        return tokens

    def map_word(
        self,        
        word: str,
    ) -> list[int]:
        """Map each bytes in a word to its corresponding value in the vocab 
        """

        if word in self.special_tokens:
            return [TokenizerTrainer.BASE_VOCAB_SIZE + self.special_tokens.index(word)]

        return list(int(byte) for byte in word.encode())

    def init_vocab(
        self,
    ) -> dict[int,bytes]:
        vocab = dict((i,i.to_bytes()) for i in range(TokenizerTrainer.BASE_VOCAB_SIZE))
        for i in range(len(self.special_tokens)):
            vocab[TokenizerTrainer.BASE_VOCAB_SIZE + i] = self.special_tokens[i].encode()

        return vocab

    def pretokenize(
        self,
        input_path: str | os.PathLike,
        boundaries: list[int],
    ) -> list[str]:
        # Setup for multithreading
        merges = []
        words = []
        args = []

        for i in range(len(boundaries)-1):
            args.append((input_path ,boundaries[i], boundaries[i+1], False))

        with Pool(processes=len(boundaries)) as pool:
            words = pool.starmap(self.pretokenize_chunk, args)

        words = [word for single_words in words for word in single_words]
        return words

    def init_attr(
        self,
        special_tokens: list[str],
        vocab_size: int,
    ):
        self.num_workers = 16
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

    def advance_token_word(
        self,
        word: list[int],
        pos: int,
    ) -> int:
        return pos + len(self.vocab[word[pos]]);

    def advance_token_id(
        self,
        id: int,
        pos: int,
    ) -> int:
        return pos + len(self.vocab[id]);

    # === train_bpe_baseline specific ===================================================================

    def get_freq(
        self,
        mapped_words: list[list[int]],
    ) -> dict[tuple[int,int], int]:
        """Get frequency count for each adjacent token pairs
        """
        freq = dict()

        for word in mapped_words:
            for pair in zip(word[:-1], word[1:]):
                if pair not in freq:
                    freq[pair] = 0
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

    def train_bpe_baseline(
        self,
        input_path: str | os.PathLike,
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

        self.init_attr(special_tokens, vocab_size)

        # Getting chunk boundaries for parallelism
        with open(input_path, "rb") as f:
            special_tokens_bytes = list(s.encode() for s in special_tokens)
            boundaries = self.find_chunk_boundaries(f, self.num_workers, "\u0003".encode());

        # Initializing vocab (all bytes & special tokens)
        vocab = self.init_vocab()
        merges = []

        # Pretokenize corpus
        words = self.pretokenize(input_path, boundaries)
        mapped_words = [self.map_word(word) for word in words]

        # Start merging adjacent pairs
        while (len(vocab) < vocab_size):
            freq = self.get_freq(mapped_words)
            try:
                max_pair = max(freq, key=lambda k: (freq[k],(vocab[k[0]], vocab[k[1]])))
            except ValueError:
                raise ValueError(f"Unable to create vocab of size {vocab_size}: Max size is {len(vocab)}")
            key = len(vocab)
            value = vocab[max_pair[0]] + vocab[max_pair[1]]
            vocab[key] = value

            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

            self.merge_pair(mapped_words, max_pair, key)

        return vocab, merges

    # === train_bpe specific ============================================================================

    def create_mapping(
        self,
        words: list[str]
    ) -> list[list[int]]:
        res = []
        for word in words:
            if word in self.special_tokens:
                res.append([TokenizerTrainer.BASE_VOCAB_SIZE + self.special_tokens.index(word) for i in range(len(word))])
            else:
                res.append(self.map_word(word))
        return res

    def init_inverted_records(
        self,
        mapping: list[list[int]]
    ) -> tuple[dict[tuple[int,int], int], dict[tuple[int,int], list[tuple[int,int]]]]:
        freq = dict()
        record = dict()

        for i in range(len(mapping)):
            word = mapping[i]
            l = 0
            r = self.advance_token_word(word, 0)
            while r < len(word):
                pair = (word[l], word[r])
                if pair not in freq:
                    freq[pair] = 0;
                    record[pair] = []
                freq[pair] += 1
                record[pair].append((i,l))


                l = self.advance_token_word(word, l)
                r = self.advance_token_word(word, r)
        
        return freq, record

    def add_pair(
        self,
        mapping,
        freq,
        record,
        i,
        l,
        r,
    ):
        n_pair = (mapping[i][l], mapping[i][r])
        if n_pair not in freq:
            freq[n_pair] = 0
            record[n_pair] = []
        freq[n_pair] += 1
        record[n_pair].append((i,l))

    def merge_record(
        self,
        mapping: list[list[int]],
        freq: dict[tuple[int,int], int],
        record: dict[tuple[int,int], list[tuple[int,int]]],
        pair: tuple[int,int],
        key: int,
    ) -> None:
        for i, l in record[pair]:
            r = self.advance_token_id(pair[0], l)
            if mapping[i][l] != pair[0] or mapping[i][r] != pair[1]:
                continue

            if (l!=0):
                freq[(mapping[i][l-1], mapping[i][l])] -= 1
            if self.advance_token_word(mapping[i], r) != len(mapping[i]):
                freq[(mapping[i][r], mapping[i][self.advance_token_word(mapping[i], r)])] -= 1

            for idx in range(l, self.advance_token_word(mapping[i], r)):
                mapping[i][idx] = key

            if (l!=0):
                prev = l - len(self.vocab[mapping[i][l-1]])
                self.add_pair(mapping, freq, record, i, prev, l)

            if self.advance_token_word(mapping[i], l) != len(mapping[i]):
                post = self.advance_token_word(mapping[i], l)
                self.add_pair(mapping, freq, record, i, l, post)

        freq.pop(pair, None)
        record.pop(pair, None)

    def train_bpe(
        self,
        input_path: str | os.PathLike,
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

        self.init_attr(special_tokens, vocab_size)

        # Getting chunk boundaries for parallelism
        with open(input_path, "rb") as f:
            special_tokens_bytes = list(s.encode() for s in special_tokens)
            boundaries = self.find_chunk_boundaries(f, self.num_workers, "\u0003".encode());

        # Initializing vocab (all bytes & special tokens)
        self.vocab = self.init_vocab()
        merges = []

        # Pretokenize corpus
        words = self.pretokenize(input_path, boundaries)

        mapping = self.create_mapping(words)
        freq, record = self.init_inverted_records(mapping)

        # Start merging adjacent pairs
        while (len(self.vocab) < vocab_size):
            try:
                max_pair = max(freq, key=lambda k: (freq[k],(self.vocab[k[0]], self.vocab[k[1]])))
            except ValueError:
                raise ValueError(f"Unable to create vocab of size {vocab_size}: Max size is {len(self.vocab)}")
            key = len(self.vocab)
            value = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.vocab[key] = value

            merges.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))

            self.merge_record(mapping, freq, record, max_pair, key)

        return self.vocab, merges



class Tokenizer():

    def encode():
        """Remember: split the chunk by special char, where special char is its own element.
        """
        raise NotImplementedError