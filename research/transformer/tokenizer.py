import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # (1) Add special tokens
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.vocab_size = 4
        
        # (2) Interate Strings
        for text in texts:
            for word in text.split():
                word = word.lower().strip()
                if self.word_to_id.get(word) is None:
                    self.word_to_id[word] = self.vocab_size
                    self.vocab_size += 1
        
        # (3) Create reverse mapping
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        ids = []
        for word in text.split():
            word = word.lower().strip()
            if self.word_to_id.get(word) is None:
                word = self.unk_token
            ids.append(self.word_to_id[word])

        return ids
     
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        words = []
        for id_ in ids:
            word = self.id_to_word.get(id_, self.unk_token)
            words.append(word)
        return ' '.join(words)


if __name__ == "__main__":
    # Example usage
    texts = [
        "Hello world",
        "This is a test",
        "Hello TensorTonic"
    ]
    
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(texts)
    
    print("Vocabulary:", tokenizer.word_to_id)
    
    sample_text = "Hello unknown world"
    encoded = tokenizer.encode(sample_text)
    print("Encoded:", encoded)
    
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

        
