import os
import io
import multiprocessing
import regex as re
import pickle

from queue import PriorityQueue
from pathlib import Path
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from typing import BinaryIO

# Helpers

class Pretokenizer():
    """Class used for pretokenizing corpus.

    Attributes:
        special_tokens: List of special tokens (tokens to keep the same for certain purposes)
        num_workers: No of default worker threads during pretokenization stage. 
    """

    # === Front-facing APIs =======================================================================

    def __init__(
        self,
        special_tokens: list[str],
        num_workers = 8,
    ):
        """Initialize the pretokenizer.

        Args:
            special_tokens: List of special tokens.
            num_workers: No of worker threads to default to for pretokenizer()
        """
        self.num_workers = num_workers
        special_tokens.sort(key = lambda x : len(x), reverse=True)
        self.special_tokens = special_tokens

    def pretokenize(
        self,
        input_path: str | os.PathLike,
        num_workers = 0,
        is_path = True,
        keep_special = False,
    ) -> list[str]:
        """Pretokenize the given corpus, return a list of pretokens.

        Args:
            input_path: Either a string or a Path to corpus.
            num_workers: No of worker threads, default to self.num_workers. 1: will not spawn threads
            is_path: If true, treat input_path as path to corpus, otherwise, treat input_path as corpus.
            keep_special: Whether to keep all special tokens delimiter or truncate. Useful for training purpose.

        Return:
            list of string of pretokens.
        """
        if num_workers == 0:
            num_workers = self.num_workers
        # Setup for multithreading
        words = []
        args = []

        if (is_path):
            input_path = Path(input_path)

        with self._open(input_path) as f:
            special_tokens_bytes = list(s.encode() for s in self.special_tokens)
            boundaries = self._find_chunk_boundaries(f, num_workers, self.CHUNK_DELIMITER.encode());


        for i in range(len(boundaries)-1):
            args.append((input_path ,boundaries[i], boundaries[i+1], keep_special))

        with Pool(processes=len(boundaries)) as pool:
            words = pool.starmap(self._pretokenize_chunk, args)

        words = [word for single_words in words for word in single_words]
        return words

    # === Constants ==================================================================================

    CHUNK_DELIMITER = "<|endoftext|>"

    # === Helpers ==================================================================================

    @staticmethod
    def _open(input_path: str | os.PathLike) -> BinaryIO:
        return open(input_path, "rb") if isinstance(input_path, os.PathLike) else io.BytesIO(input_path.encode())

    def _find_chunk_boundaries(
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

    def _pretokenize_chunk(
        self,
        file_path: str | os.PathLike,
        start: int,
        end: int,
        keep_special: bool = True,
    ) -> list[str]:
        """Process a chunk of data in file, outlined by [start, end)

        This function either strips all special tokens, or keep them as a monolithic element

        """
        # Splitting on special character
        with self._open(file_path) as file:
            if keep_special:
                split_pat = re.compile(
                    "(" + "|".join(re.escape(special_token) for special_token in self.special_tokens) + ")"
                )
            else:
                split_pat = re.compile(
                    "|".join(re.escape(special_token) for special_token in self.special_tokens)
                )

            file.seek(start, os.SEEK_SET)
            corpus = file.read(end-start).decode('utf-8', errors='ignore')
            words = (list(re.split(split_pat, corpus)) if len(self.special_tokens) else [corpus])

            # Pretokenization
            tokens = []
            for word in words:
                if word in self.special_tokens:
                    tokens.append(word)
                else:
                    tokens += re.findall(TokenizerTrainer.PAT, word)

        return tokens

class TokenizerTrainer():
    """Train a tokenizer based on some corpus

    This classes uses BPE tokenizer.

    Attributes:
        - special_tokens : List of special tokens
        - vocab_size : Targetted output vocab size
        - num_workers : No of workers to use for pretokenization, hardcoded 16

    """

    # === Constants ====================================================================================

    PAT = PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    BASE_VOCAB_SIZE = 256
    CHUNK_DELIMITER = "<|endoftext|>"

    # === Front facing APIs ==========================================================================

    def __init__(
        self,
        vocab_size: int,
        special_tokens: list[str],
    ) -> None:
        """Initialize the Tokenizer Trainer

        Args:
            - special_tokens : List of special tokens
            - vocab_size : Targetted output vocab size
        """
        self._init_attr(special_tokens, vocab_size)

    def train_bpe_baseline(
        self,
        input_path: str | os.PathLike,
    ) -> tuple[
        dict[int,bytes],
        list[tuple[bytes, bytes]],
    ]:
        """Train Byte-pair Encoder, given corpus path & targeted vocab_size

        Inefficient implementation. Serves as a baseline for the more efficient impl.

        Args:
            - input_path : Path to corpus
            - vocab_size : Targetted output vocab size
            - special_tokens : List of special tokens

        Returns:
            - vocab: dict[int, bytes] : The resulting vocab after BPE merge
            - merges: list[tuple[bytes, bytes]] : List of BPE merges produces from training
        
        """
        # Initializing vocab (all bytes & special tokens)
        vocab = self._init_vocab()
        merges = []

        # Pretokenize corpus
        pretokenizer = Pretokenizer(self.special_tokens)
        words = pretokenizer.pretokenize(input_path)
        mapped_words = [self._map_word(word) for word in words]

        # Start merging adjacent pairs
        while (len(vocab) < self.vocab_size):
            freq = self._get_freq(mapped_words)
            try:
                max_pair = max(freq, key=lambda k: (freq[k],(vocab[k[0]], vocab[k[1]])))
            except ValueError:
                raise ValueError(f"Unable to create vocab of size {self.vocab_size}: Max size is {len(vocab)}")
            key = len(vocab)
            value = vocab[max_pair[0]] + vocab[max_pair[1]]
            vocab[key] = value

            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))

            self._merge_pair(mapped_words, max_pair, key)

        return vocab, merges

    def train_bpe(
        self,
        input_path: str | os.PathLike,
    ) -> tuple[
        dict[int,bytes],
        list[tuple[bytes, bytes]],
    ]:
        """Train Byte-pair Encoder, given corpus path & targeted vocab_size

        Efficient implementation. Use this one for most cases.
        
        Args:
            - input_path : Path to corpus
            - vocab_size : Targetted output vocab size
            - special_tokens : List of special tokens

        Returns:
            - vocab: dict[int, bytes] : The resulting vocab after BPE merge
            - merges: list[tuple[bytes, bytes]] : List of BPE merges produces from training
        
        """

        # Initializing vocab (all bytes & special tokens)
        self.vocab = self._init_vocab()
        merges = []

        # Pretokenize corpus
        pretokenizer = Pretokenizer(self.special_tokens)
        words = pretokenizer.pretokenize(input_path)

        mapping = self._create_mapping(words)
        freq, record = self._init_inverted_records(mapping)

        # Start merging adjacent pairs
        while (len(self.vocab) < self.vocab_size):
            try:
                max_pair = max(freq, key=lambda k: (freq[k],(self.vocab[k[0]], self.vocab[k[1]])))
            except ValueError:
                raise ValueError(f"Unable to create vocab of size {self.vocab_size}: Max size is {len(self.vocab)}")
            key = len(self.vocab)
            value = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]
            self.vocab[key] = value

            merges.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))

            self._merge_record(mapping, freq, record, max_pair, key)

        return self.vocab, merges

    # === General helper functions ==================================================================

    def _map_word(
        self,        
        word: str,
    ) -> list[int]:
        """Map each bytes in a word to its corresponding value in the vocab 
        """

        if word in self.special_tokens:
            return [TokenizerTrainer.BASE_VOCAB_SIZE + self.special_tokens.index(word)]

        return list(int(byte) for byte in word.encode())

    def _init_vocab(
        self,
    ) -> dict[int,bytes]:
        vocab = dict((i,i.to_bytes()) for i in range(TokenizerTrainer.BASE_VOCAB_SIZE))
        for i in range(len(self.special_tokens)):
            vocab[TokenizerTrainer.BASE_VOCAB_SIZE + i] = self.special_tokens[i].encode()

        return vocab

    def _init_attr(
        self,
        special_tokens: list[str],
        vocab_size: int,
    ):
        self.num_workers = 16
        self.special_tokens = special_tokens
        self.vocab_size = vocab_size

    def _advance_token_word(
        self,
        word: list[int],
        pos: int,
    ) -> int:
        return pos + len(self.vocab[word[pos]]);

    def _advance_token_id(
        self,
        id: int,
        pos: int,
    ) -> int:
        return pos + len(self.vocab[id]);
    
    # === train_bpe_baseline specific helpers ===========================================================

    def _get_freq(
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

    def _merge_pair(
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

    # === train_bpe specific helpers ======================================================================

    def _create_mapping(
        self,
        words: list[str]
    ) -> list[list[int]]:
        res = []
        for word in words:
            if word in self.special_tokens:
                res.append([TokenizerTrainer.BASE_VOCAB_SIZE + self.special_tokens.index(word) for i in range(len(word.encode()))])
            else:
                res.append(self._map_word(word))
        return res

    def _init_inverted_records(
        self,
        mapping: list[list[int]]
    ) -> tuple[dict[tuple[int,int], int], dict[tuple[int,int], list[tuple[int,int]]]]:
        freq = dict()
        record = dict()

        for i in range(len(mapping)):
            word = mapping[i]
            l = 0
            r = self._advance_token_word(word, 0)
            while r < len(word):
                pair = (word[l], word[r])
                if pair not in freq:
                    freq[pair] = 0;
                    record[pair] = []
                freq[pair] += 1
                record[pair].append((i,l))


                l = self._advance_token_word(word, l)
                r = self._advance_token_word(word, r)
        
        return freq, record

    def _add_pair(
        self,
        mapping,
        freq,
        record,
        i, l, r,
    ):
        n_pair = (mapping[i][l], mapping[i][r])
        if n_pair not in freq:
            freq[n_pair] = 0
            record[n_pair] = []
        freq[n_pair] += 1
        record[n_pair].append((i,l))

    def _merge_record(
        self,
        mapping: list[list[int]],
        freq: dict[tuple[int,int], int],
        record: dict[tuple[int,int], list[tuple[int,int]]],
        pair: tuple[int,int],
        key: int,
    ) -> None:
        for i, l in record[pair]:
            r = self._advance_token_id(pair[0], l)
            if mapping[i][l] != pair[0] or mapping[i][r] != pair[1]:
                continue

            if (l!=0):
                freq[(mapping[i][l-1], mapping[i][l])] -= 1
            if self._advance_token_word(mapping[i], r) != len(mapping[i]):
                freq[(mapping[i][r], mapping[i][self._advance_token_word(mapping[i], r)])] -= 1

            for idx in range(l, self._advance_token_word(mapping[i], r)):
                mapping[i][idx] = key

            if (l!=0):
                prev = l - len(self.vocab[mapping[i][l-1]])
                self._add_pair(mapping, freq, record, i, prev, l)

            if self._advance_token_word(mapping[i], l) != len(mapping[i]):
                post = self._advance_token_word(mapping[i], l)
                self._add_pair(mapping, freq, record, i, l, post)

        freq.pop(pair, None)
        record.pop(pair, None)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """API Wrapper for train_bpe
    """
    trainer = TokenizerTrainer(vocab_size, special_tokens)
    return trainer.train_bpe(input_path)

class Tokenizer():

    # === Front facing APIs ======================================================================================

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.inv_vocab = {}
        for key in vocab:
            self.inv_vocab[vocab[key]] = key

        self.special_tokens = special_tokens if special_tokens != None else []
        for special_token in self.special_tokens:
            encoded_token = special_token.encode()
            if encoded_token not in self.inv_vocab:
                key = len(vocab)
                self.vocab[key] = encoded_token
                self.inv_vocab[encoded_token] = key

        self.merges = {
            (self.inv_vocab[merges[i][0]], self.inv_vocab[merges[i][1]]):
            (i, self.inv_vocab[merges[i][0] + merges[i][1]])
        for i in range(len(merges))}

        self.pretokenizer = Pretokenizer(self.special_tokens)

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """Remember: split the chunk by special char, where special char is its own element.
        """

        words = self.pretokenizer.pretokenize(text, 1, False, True)
        mapping = self._create_mapping(words)
        records, heap = self._init_inverted_records(mapping)
        
        while not heap.empty():
            _, pair = heap.get()
            if pair in self.merges and pair in records:
                key = self.merges[pair][1]
                self._merge_record(mapping, records, heap, pair, key)

        compressed = self._compress_mapping(mapping)
        return [key for word in compressed for key in word]

    def encode_iterable(
        self,
        iterable: Iterable[str]
    ) -> Iterator[int]:
        return Tokenizer.TokenIterator(iterable, self)

    def decode(
        self,
        ids: list[int]
    ) -> str:
        res = bytearray()
        for id in ids:
            res.extend(self.vocab[id])
        return bytes(res).decode(encoding='utf-8', errors='replace')

    # === Static Helpers ====================================================================

    @staticmethod
    def dump_vocab(
        vocab: dict[int, bytes],
        vocab_filepath: str | os.PathLike,
    ) -> None:
        with open(vocab_filepath, 'wb') as file:
            pickle.dump(vocab, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def dump_merges(
        merges: list[tuple[bytes, bytes]],
        merges_filepath: str | os.PathLike,
    ) -> None:
        with open(merges_filepath, 'wb') as file:
            pickle.dump(merges, file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def find_chunk_boundaries(
        file: BinaryIO,
        desired_chunk_size: int,
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

        num_chunks = file_size // desired_chunk_size

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * desired_chunk_size for i in range(num_chunks + 1)]
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

    # === Instantiation =================================================================================

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)
        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)

    # === Token encoding private helpers =====================================================================

    def _map_word(
        self,        
        word: str,
    ) -> list[int]:
        """Map each bytes in a word to its corresponding value in the vocab 
        """

        if word in self.special_tokens:
            return [self.inv_vocab[word.encode()]]

        return [self.inv_vocab[byte.to_bytes()] for byte in word.encode()]

    def _advance_token_word(
        self,
        word: list[int],
        pos: int,
    ) -> int:
        return pos + len(self.vocab[word[pos]]);

    def _advance_token_id(
        self,
        id: int,
        pos: int,
    ) -> int:
        return pos + len(self.vocab[id]);

    def _create_mapping(
        self,
        words: list[str]
    ) -> list[list[int]]:
        res = []
        for word in words:
            if word in self.special_tokens:
                res.append([self.inv_vocab[word.encode()] for i in range(len(word.encode()))])
            else:
                res.append(self._map_word(word))
        return res

    def _init_inverted_records(
        self,
        mapping: list[list[int]]
    ) -> tuple[dict[tuple[int,int], list[tuple[int,int]]], PriorityQueue]:
        record = dict()
        heap = PriorityQueue()

        for i in range(len(mapping)):
            word = mapping[i]
            l = 0
            r = self._advance_token_word(word, 0)
            while r < len(word):
                pair = (word[l], word[r])

                if pair in self.merges:
                    if pair not in record:
                        record[pair] = []
                    record[pair].append((i,l))
                    heap.put((self.merges[pair][0], pair))

                l = self._advance_token_word(word, l)
                r = self._advance_token_word(word, r)
        
        return record, heap

    def _add_pair(
        self,
        mapping: list[list[int]],
        record: dict[tuple[int,int], list[tuple[int,int]]],
        heap: PriorityQueue,
        i: int,
        l: int,
        r: int,
    ):
        n_pair = (mapping[i][l], mapping[i][r])

        if n_pair in self.merges:
            if n_pair not in record:
                record[n_pair] = []
            record[n_pair].append((i,l))
            heap.put((self.merges[n_pair][0], n_pair))

    def _merge_record(
        self,
        mapping: list[list[int]],
        record: dict[tuple[int,int], list[tuple[int,int]]],
        heap: PriorityQueue,
        pair: tuple[int,int],
        key: int,
    ) -> None:
        for i, l in record[pair]:
            r = self._advance_token_id(pair[0], l)
            if mapping[i][l] != pair[0] or mapping[i][r] != pair[1]:
                continue

            for idx in range(l, self._advance_token_word(mapping[i], r)):
                mapping[i][idx] = key

            if (l!=0):
                prev = l - len(self.vocab[mapping[i][l-1]])
                self._add_pair(mapping, record, heap, i, prev, l)

            if self._advance_token_word(mapping[i], l) != len(mapping[i]):
                post = self._advance_token_word(mapping[i], l)
                self._add_pair(mapping, record, heap, i, l, post)

        record.pop(pair, None)

    def _compress_mapping(
        self,
        mapping: list[list[int]],
    ) -> list[list[int]]:
        res = []
        for word in mapping:
            res.append([])
            ptr = 0
            while ptr < len(word):
                res[-1].append(word[ptr])
                ptr = self._advance_token_word(word, ptr)
        return res
    
    # === Token iterator class ===============================================================================

    class TokenIterator():
        def _advance_str(
            self,
        ):
            self.ptr = 0
            try:
                nxt_string = next(self.iterable)
            except StopIteration:
                self.encoded = None
                return
            self.encoded = self.tokenizer.encode(nxt_string)

        def __init__(
            self,
            iterable: Iterable[str],
            tokenizer: "Tokenizer",
        ):
            self.iterable = iter(iterable)
            self.tokenizer = tokenizer
            self.ptr = 0
            self.encoded = []
            while self.encoded != None and self.ptr == len(self.encoded):
                self._advance_str()

        def __next__(
            self,
        ):
            while self.encoded != None and self.ptr == len(self.encoded):
                self._advance_str()
            if self.encoded == None:
                raise StopIteration
            key = self.encoded[self.ptr]
            self.ptr += 1
            return key
            
        def __iter__(
            self,
        ):
            return self
