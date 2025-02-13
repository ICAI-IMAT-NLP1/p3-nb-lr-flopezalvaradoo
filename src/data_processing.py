from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    # Open the file, go line by line, separate sentence and label, tokenize the sentence and create SentimentExample object
    examples: List[SentimentExample] = []
    
    with open(infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            parts = line.split("\t")
            
            sentence, label = parts
            tokens = tokenize(sentence)  
            label_int = int(label)
            examples.append(SentimentExample(words=tokens, label=label_int))
    return examples


def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    
    word_counter = Counter()

    for example in examples:
        word_counter.update(example.words)

    vocab = {word: idx for idx, word in enumerate(word_counter.keys())}
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    vocab_size = len(vocab)
    # Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(vocab_size)
    vocab_words = vocab.keys()
    for word in text:
        if word not in vocab_words:
            continue
        idx = vocab[word]
        
        if binary:
            bow[idx] = 1
        else: 
            bow[idx] += 1
    return bow
