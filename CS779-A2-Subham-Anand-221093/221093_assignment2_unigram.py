import argparse
import os
import math
import heapq
import time
from collections import defaultdict

# A very small number in log-space to represent log(0), preventing math errors.
LOG_PROB_ZERO = -1e18
SPACE_SYMBOL = '_'


class TrieNode:
    """Represents a single node in the prefix tree."""
    def __init__(self):
        self.children = {}
        # Stores the log probability if this node marks the end of a token.
        self.log_prob = None

class VocabularyTrie:
    """A prefix tree that stores the vocabulary for fast token matching."""
    def __init__(self, vocabulary=None):
        self.root = TrieNode()
        if vocabulary:
            for token, log_prob in vocabulary.items():
                self.add_token(token, log_prob)

    def add_token(self, token, log_prob):
        """Adds a token and its log probability to the trie."""
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.log_prob = log_prob

def read_corpus_from_path(path):
    """
    Loads text from a file or all files in a directory.
    
    This function normalizes all whitespace into single spaces and then
    replaces them with the SPACE_SYMBOL for consistent processing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified training path was not found: {path}")

    full_text = ""
    if os.path.isdir(path):
        print(f"Reading all files from directory: {path}")
        for entry in sorted(os.listdir(path)):
            full_path = os.path.join(path, entry)
            if os.path.isfile(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    full_text += f.read()
    else: # It's a file
        print(f"Reading from file: {path}")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
    
    # Normalize whitespace and replace with the designated symbol
    normalized = ' '.join(full_text.split())
    return normalized.replace(' ', SPACE_SYMBOL)

def segment_text(text, model_source):
    """
    Finds the most likely token segmentation of a text using the Viterbi algorithm.
    
    This function can work with two types of model sources for flexibility:
    1. A VocabularyTrie: Optimized for segmenting large texts during the E-step.
    2. A dict: Simpler and efficient for segmenting short strings, like in the loss calculation step.
    """
    text_len = len(text)
    # log_likelihoods[i] stores the max log-probability of any segmentation for text[:i]
    log_likelihoods = [LOG_PROB_ZERO] * (text_len + 1)
    log_likelihoods[0] = 0
    # best_prev_indices[i] stores the start index of the last token in the best segmentation of text[:i]
    best_prev_indices = [None] * (text_len + 1)

    # Use the appropriate lookup method based on the model type
    if isinstance(model_source, VocabularyTrie):
        # Trie-based segmentation (fast for long texts)
        for i in range(text_len):
            if log_likelihoods[i] == LOG_PROB_ZERO:
                continue
            
            node = model_source.root
            for j in range(i, text_len):
                char = text[j]
                if char not in node.children:
                    break # No more valid tokens with this prefix
                node = node.children[char]
                if node.log_prob is not None:
                    # Found a valid token: text[i:j+1]
                    current_log_prob = log_likelihoods[i] + node.log_prob
                    end_pos = j + 1
                    if current_log_prob > log_likelihoods[end_pos]:
                        log_likelihoods[end_pos] = current_log_prob
                        best_prev_indices[end_pos] = i
    else: # Dictionary-based segmentation
        log_probs_dict = model_source
        for i in range(1, text_len + 1):
            # Check for tokens ending at position i. Limit search to a max token length (e.g., 12).
            for j in range(max(0, i - 12), i):
                token = text[j:i]
                token_log_prob = log_probs_dict.get(token)
                if token_log_prob is not None:
                    current_log_prob = log_likelihoods[j] + token_log_prob
                    if current_log_prob > log_likelihoods[i]:
                        log_likelihoods[i] = current_log_prob
                        best_prev_indices[i] = j

    # If the end is unreachable, segmentation failed.
    if log_likelihoods[text_len] == LOG_PROB_ZERO:
        return [], LOG_PROB_ZERO

    # Backtrack from the end to reconstruct the best token sequence.
    tokens = []
    current_pos = text_len
    while current_pos > 0:
        prev_pos = best_prev_indices[current_pos]
        if prev_pos is None: # Fallback for characters not in vocabulary
            tokens.append(text[current_pos - 1:current_pos])
            current_pos -= 1
        else:
            tokens.append(text[prev_pos:current_pos])
            current_pos = prev_pos
    
    tokens.reverse()
    return tokens, log_likelihoods[text_len]


class UnigramTokenizer:
    """
    An implementation of the Unigram Segmentation algorithm for tokenization.
    
    This class encapsulates the training, tokenization, and detokenization logic.
    The training process uses an Expectation-Maximization (EM) approach to
    iteratively refine a vocabulary from a large seed set down to a desired size.
    """
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.model = {} # The vocabulary: maps tokens to their log probabilities

    def train(self, corpus):
        """
        Trains the tokenizer model on a given text corpus.
        """
        # 1. Initialize a large "seed" vocabulary from characters and frequent substrings.
        self._initialize_seed_vocab(corpus)
        
        # 2. Iteratively run the EM algorithm to refine and prune the vocabulary.
        while len(self.model) > self.vocab_size:
            # print(f"\n--- Running EM cycle. Current vocab size: {len(self.model)} ---")
            
            # E-Step: Segment the corpus to get expected token counts.
            token_counts = self._e_step(corpus)

            # M-Step: Update model probabilities based on new counts.
            self._m_step(token_counts)

            # Pruning Step: Calculate the "loss" for each token and remove the least useful ones.
            self._prune_vocabulary()
        
        self.model = {token: prob for token, prob in sorted(self.model.items())}

    def _initialize_seed_vocab(self, corpus):
        """Creates the initial large vocabulary."""
        # All individual characters are essential and form the base.
        char_counts = defaultdict(int)
        for char in corpus:
            char_counts[char] += 1
        self.essential_tokens = set(char_counts.keys())
        
        # Add frequent substrings to the seed vocabulary.
        substring_counts = defaultdict(int)
        for i in range(len(corpus)):
            for j in range(i + 2, min(i + 10, len(corpus) + 1)):
                substring_counts[corpus[i:j]] += 1
        
        seed_vocab = set(self.essential_tokens)
        
        # Sort substrings by frequency and add the most common ones.
        sorted_substrings = sorted(substring_counts.items(), key=lambda item: item[1], reverse=True)
        
        for sub, count in sorted_substrings:
            if len(seed_vocab) >= 20000: # Limit initial seed size
                break
            if count >= 2:
                seed_vocab.add(sub)
        
        # Assign initial, rough log probabilities based on counts.
        total_count = float(sum(char_counts.values()) + sum(substring_counts.values()))
        log_total = math.log(total_count)
        self.model = {
            token: math.log(char_counts.get(token, substring_counts.get(token, 1.0))) - log_total
            for token in seed_vocab
        }

    def _e_step(self, corpus):
        """
        The "Expectation" step: segment the entire corpus with the current model
        to determine the expected frequency of each token.
        """
        # A Trie is built for this step for maximum segmentation speed.
        model_trie = VocabularyTrie(self.model)
        
        token_counts = defaultdict(int)
        # Process the corpus in chunks to manage memory.
        chunk_size = 100000
        for i in range(0, len(corpus), chunk_size):
            chunk = corpus[i : i + chunk_size]
            tokens, _ = segment_text(chunk, model_trie)
            for token in tokens:
                token_counts[token] += 1
        return token_counts
    
    def _m_step(self, token_counts):
        """
        The "Maximization" step: recalculate the log probability of each
        token based on its count from the E-step.
        """
        total_token_count = sum(token_counts.values())
        if total_token_count == 0:
            print("Warning: M-step found no tokens. Halting.")
            return

        log_total = math.log(total_token_count)
        for token in self.model:
            # Use a small count (1.0) for tokens that didn't appear (smoothing).
            count = token_counts.get(token, 1.0)
            self.model[token] = math.log(count) - log_total

    def _prune_vocabulary(self):
        """
        Calculates how "costly" it would be to remove each token and then
        prunes the least costly ones to reduce the vocabulary size.
        """
        token_losses = []
        candidates_for_pruning = [t for t in self.model if t not in self.essential_tokens]

        for token in candidates_for_pruning:
            # To calculate the loss, temporarily remove the token from the model.
            original_log_prob = self.model.pop(token)
            
            # The loss is the drop in likelihood when the token is segmented
            # using the rest of the vocabulary. We use the dict-based segmenter here
            # because the strings being segmented (the tokens themselves) are very short.
            _, alternative_log_prob = segment_text(token, self.model)
            
            # Restore the token to the model.
            self.model[token] = original_log_prob
            
            # Loss = P(token) / P(alternative_segmentation)
            # In log space, this is log(P(token)) - log(P(alt))
            loss = original_log_prob - alternative_log_prob
            token_losses.append((loss, token))
        
        # Determine how many tokens to remove in this pass.
        current_size = len(self.model)
        gap_to_target = current_size - self.vocab_size
        if gap_to_target <= 0:
            return
            
        # Prune a significant fraction of the candidates or the gap to speed up convergence.
        prune_target_count = max(
            int(len(token_losses) * 0.10),      # Prune at least 10% of candidates
            int(gap_to_target * 0.40)          # Aim to close 40% of the gap to the target
        )
        num_to_prune = min(prune_target_count, gap_to_target)

        # Find the tokens with the smallest loss.
        tokens_to_remove = heapq.nsmallest(num_to_prune, token_losses, key=lambda x: x[0])
        # print(f"Pruning {len(tokens_to_remove)} tokens with the lowest loss.")

        for _, token in tokens_to_remove:
            if token in self.model:
                del self.model[token]

    def tokenize(self, text):
        """Tokenizes a new piece of text using the trained model."""
        normalized = ' '.join(text.split()).replace(' ', SPACE_SYMBOL)
        model_trie = VocabularyTrie(self.model)
        tokens, _ = segment_text(normalized, model_trie)
        return tokens
    
    def detokenize(self, tokens):
        """Reconstructs the original text from a list of tokens."""
        return "".join(tokens).replace(SPACE_SYMBOL, ' ').strip()
        
    def get_vocabulary(self):
        """Returns the list of tokens in the final vocabulary."""
        return list(self.model.keys())

    def save_vocabulary_file(self, rollno):
        """Saves the final vocabulary to a text file."""
        filename = f"{rollno}_assignment2_unigram_vocab_{self.vocab_size}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for token in self.get_vocabulary():
                f.write(token + "\n")
        print(f"Vocabulary saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        # formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--train", type=str, required=True, help="Path to the training data file or directory.")
    parser.add_argument("--input", type=str, required=True, help="Path to a sample text file to tokenize.")
    parser.add_argument("--vocab_size", type=int, required=True, help="The desired final size of the vocabulary.")
    args = parser.parse_args()
    
    rollno = "221093" 

    # Load Data 
    print("--- Step 1: Loading Corpus ---")
    corpus = read_corpus_from_path(args.train)
    print(f"Corpus loaded successfully ({len(corpus):,} characters).\n")

    # train 
    print("--- Step 2: Training Tokenizer ---")
    start_time = time.perf_counter()
    tokenizer = UnigramTokenizer(vocab_size=args.vocab_size)
    tokenizer.train(corpus)
    end_time = time.perf_counter()
    duration_minutes = (end_time - start_time) / 60
    print(f"\nTraining finished in {duration_minutes:.2f} minutes.")
    print(f"Final vocabulary size is {len(tokenizer.get_vocabulary())}.")
    
    # save vocab
    print("\n--- Step 3: Saving Model Vocabulary ---")
    tokenizer.save_vocabulary_file(rollno)

    # tokenize
    print("\n--- Step 4: Tokenizing Sample Text ---")
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    
    tokens = tokenizer.tokenize(sample_text)
    
    token_filename = f"{rollno}_assignment2_unigram_tokens.txt"
    with open(token_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))
    print(f"Tokenized output saved to {token_filename}")

    # detokenize
    print("\n--- Step 5: Detokenizing and Saving Result ---")
    detokenized_text = tokenizer.detokenize(tokens)
    
    detok_filename = f"{rollno}_assignment2_unigram_detokenized.txt"
    with open(detok_filename, "w", encoding="utf-8") as f:
        f.write(detokenized_text)
    print(f"Detokenized text saved to {detok_filename}")
    

