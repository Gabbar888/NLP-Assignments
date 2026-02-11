import argparse
import unicodedata
import math
import sys
import os
from collections import Counter, defaultdict
import heapq

# Configuration for special tokens and subword prefixes.
SUBWORD_PREFIX = "##"
SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]

def prepare_text(text_string):
    """Normalizes a string using NFKC."""
    return unicodedata.normalize("NFKC", text_string)

def split_by_whitespace(text_line):
    """Splits a line into tokens based on whitespace."""
    return text_line.strip().split()

class TrieNode:
    """A node in the Trie data structure."""
    __slots__ = ("children", "is_end_of_token")
    def __init__(self):
        self.children = {}
        self.is_end_of_token = False

class TokenTrie:
    """Trie for efficient longest-prefix matching of tokens."""
    def __init__(self):
        self.root = TrieNode()

    def add(self, token):
        """Adds a token to the trie."""
        node = self.root
        for char in token:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_token = True

    def find_longest_match(self, text, start_index=0):
        """Finds the longest token in the trie that is a prefix of the text."""
        node = self.root
        longest_match_len = 0
        current_len = 0
        i = start_index
        while i < len(text):
            char = text[i]
            if char in node.children:
                node = node.children[char]
                current_len += 1
                if node.is_end_of_token:
                    longest_match_len = current_len
                i += 1
            else:
                break
        
        if longest_match_len == 0:
            return 0, None
        return longest_match_len, text[start_index : start_index + longest_match_len]

class WordPieceTrainer:
    """
    Manages the WordPiece vocabulary training process using a node pool
    to efficiently track token pairs and merges.
    """
    def __init__(self, vocab_size, verbose=False):
        self.desired_vocab_size = int(vocab_size)
        self.verbose = verbose
        
        # Token mapping
        self.id_to_token = []
        self.token_to_id = {}
        for token in SPECIAL_TOKENS:
            self._get_or_create_token_id(token)

        # Word and node pool data structures
        self.word_to_idx = {}
        self.word_counts = []
        self.node_token_id = []
        self.node_next = []
        self.node_prev = []
        self.node_word_idx = []
        self.node_char_pos = []
        self.is_node_dead = []
        self.word_start_node = []
        
        # Pair statistics for merging
        self.pair_frequencies = Counter()
        self.pair_locations = defaultdict(list)
        self.token_adjacencies = defaultdict(set)
        self.merge_queue = []
        self.merge_history = []
        self.total_pair_count = 0
        
        # For final tokenization
        self.start_tokens = set()

    def _get_or_create_token_id(self, token_str):
        """Adds a new token string to the vocabulary, returning its ID."""
        if token_str in self.token_to_id:
            return self.token_to_id[token_str]
        token_id = len(self.id_to_token)
        self.id_to_token.append(token_str)
        self.token_to_id[token_str] = token_id
        return token_id

    def load_corpus_and_initialize(self, corpus_path):
        """Reads a corpus, counts word frequencies, and builds the initial node pool."""
        word_frequencies = Counter()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                normalized_line = prepare_text(line)
                if not normalized_line:
                    continue
                for word in split_by_whitespace(normalized_line):
                    word_frequencies[word] += 1
        
        for word, count in sorted(word_frequencies.items()):
            word_idx = len(self.word_counts)
            self.word_to_idx[word] = word_idx
            self.word_counts.append(count)
            
            head_node = -1
            previous_node_idx = -1
            current_word_node_indices = []
            
            for char_pos, char in enumerate(word):
                token_id = self._get_or_create_token_id(char)
                node_idx = len(self.node_token_id)
                
                self.node_token_id.append(token_id)
                self.node_next.append(-1)
                self.node_prev.append(previous_node_idx)
                self.node_word_idx.append(word_idx)
                self.node_char_pos.append(char_pos)
                self.is_node_dead.append(False)
                
                if previous_node_idx != -1:
                    self.node_next[previous_node_idx] = node_idx
                else:
                    head_node = node_idx
                
                previous_node_idx = node_idx
                current_word_node_indices.append(node_idx)
                
            self.word_start_node.append(head_node)
            self.total_pair_count += (len(current_word_node_indices) -1) * count if len(current_word_node_indices) > 1 else 0

            for i in range(len(current_word_node_indices) - 1):
                left_node_idx = current_word_node_indices[i]
                right_node_idx = current_word_node_indices[i+1]
                id1 = self.node_token_id[left_node_idx]
                id2 = self.node_token_id[right_node_idx]
                
                self.pair_frequencies[(id1, id2)] += count
                self.pair_locations[(id1, id2)].append(left_node_idx)
                self.token_adjacencies[id1].add(id2)
                self.token_adjacencies[id2].add(id1)

        if self.verbose:
            print(f"Initialization complete. Words: {len(self.word_counts)}, Tokens: {len(self.id_to_token)}, N: {self.total_pair_count}", file=sys.stderr)

    def _calculate_pair_score(self, token_pair):
        """Calculates the score for merging a pair of tokens."""
        pair_freq = self.pair_frequencies.get(token_pair, 0)
        if pair_freq <= 0 or self.total_pair_count <= 0:
            return float("-inf")
        # The score is based on the log-likelihood of the pair vs individual tokens.
        return pair_freq * (math.log(self.total_pair_count) - math.log(pair_freq))

    def _enqueue_pair(self, id1, id2):
        """Calculates the score for a pair and pushes it to the priority queue."""
        token_pair = (id1, id2)
        freq = self.pair_frequencies.get(token_pair, 0)
        if freq <= 0:
            return
        
        score = self._calculate_pair_score(token_pair)
        str1 = self.id_to_token[id1]
        str2 = self.id_to_token[id2]
        # Use a max-heap by negating the score.
        heapq.heappush(self.merge_queue, (-score, str1, str2, id1, id2))

    def _build_initial_queue(self):
        """Initializes the merge queue with all existing token pairs."""
        self.merge_queue = []
        # Sort for deterministic behavior
        sorted_pairs = sorted(list(self.pair_frequencies.items()), key=lambda kv: (self.id_to_token[kv[0][0]], self.id_to_token[kv[0][1]]))
        for (id1, id2), count in sorted_pairs:
            self._enqueue_pair(id1, id2)

    def run_training(self):
        """Performs iterative merging of token pairs to build the vocabulary."""
        initial_vocab_size = len(self.id_to_token)
        num_merges_required = self.desired_vocab_size - initial_vocab_size
        
        if num_merges_required <= 0:
            if self.verbose:
                print("Target vocab size already met. No merges needed.", file=sys.stderr)
            return

        if self.verbose:
            print(f"Starting training. Merges to perform: {num_merges_required}", file=sys.stderr)
        
        self._build_initial_queue()
        merges_completed = 0
        while merges_completed < num_merges_required and self.merge_queue:
            neg_score, _, _, id1, id2 = heapq.heappop(self.merge_queue)
            
            token_pair = (id1, id2)
            if self.pair_frequencies.get(token_pair, 0) <= 0:
                continue
            
            # Re-calculate score to check for staleness
            current_score = self._calculate_pair_score(token_pair)
            if abs(-neg_score - current_score) > 1e-11:
                heapq.heappush(self.merge_queue, (-current_score, self.id_to_token[id1], self.id_to_token[id2], id1, id2))
                continue
            
            merged_token_str = self.id_to_token[id1] + self.id_to_token[id2]
            merged_id = self._get_or_create_token_id(merged_token_str)
            self.merge_history.append(merged_token_str)
            
            locations = self.pair_locations.get(token_pair, [])
            # Sort for deterministic updates.
            sorted_locations = sorted(locations, key=lambda node: (self.node_word_idx[node], self.node_char_pos[node]))

            for left_node in sorted_locations:
                if self.is_node_dead[left_node] or self.node_token_id[left_node] != id1:
                    continue
                
                right_node = self.node_next[left_node]
                if right_node == -1 or self.is_node_dead[right_node] or self.node_token_id[right_node] != id2:
                    continue
                
                word_idx = self.node_word_idx[left_node]
                word_freq = self.word_counts[word_idx]
                
                prev_node = self.node_prev[left_node]
                next_node = self.node_next[right_node]
                
                # Update counts for affected pairs
                self.pair_frequencies[token_pair] -= word_freq
                if prev_node != -1:
                    self.pair_frequencies[(self.node_token_id[prev_node], id1)] -= word_freq
                if next_node != -1:
                    self.pair_frequencies[(id2, self.node_token_id[next_node])] -= word_freq

                # Perform the merge in the node list
                self.node_token_id[left_node] = merged_id
                self.is_node_dead[right_node] = True
                self.node_next[left_node] = next_node
                if next_node != -1:
                    self.node_prev[next_node] = left_node
                
                self.total_pair_count -= word_freq

                # Create new pairs and update their counts
                if prev_node != -1:
                    prev_id = self.node_token_id[prev_node]
                    new_pair = (prev_id, merged_id)
                    self.pair_frequencies[new_pair] += word_freq
                    self.pair_locations[new_pair].append(prev_node)
                    self.token_adjacencies[prev_id].add(merged_id)
                    self.token_adjacencies[merged_id].add(prev_id)
                if next_node != -1:
                    next_id = self.node_token_id[next_node]
                    new_pair = (merged_id, next_id)
                    self.pair_frequencies[new_pair] += word_freq
                    self.pair_locations[new_pair].append(left_node)
                    self.token_adjacencies[merged_id].add(next_id)
                    self.token_adjacencies[next_id].add(merged_id)

            self.pair_locations.pop(token_pair, None)
            
            # Find all potentially affected pairs and update them in the queue
            affected_neighbors = set()
            affected_neighbors.update(self.token_adjacencies.get(merged_id, set()))
            affected_neighbors.update(self.token_adjacencies.get(id1, set()))
            affected_neighbors.update(self.token_adjacencies.get(id2, set()))
            
            candidate_pairs = []
            for neighbor_id in affected_neighbors:
                if self.pair_frequencies.get((merged_id, neighbor_id), 0) > 0:
                    candidate_pairs.append((merged_id, neighbor_id))
                if self.pair_frequencies.get((neighbor_id, merged_id), 0) > 0:
                    candidate_pairs.append((neighbor_id, merged_id))
            
            sorted_candidates = sorted(candidate_pairs, key=lambda p: (self.id_to_token[p[0]], self.id_to_token[p[1]]))
            for (cand_id1, cand_id2) in sorted_candidates:
                self._enqueue_pair(cand_id1, cand_id2)
            
            merges_completed += 1
        
        if self.verbose:
            print(f"Training finished. Merges: {merges_completed}. Vocab size: {len(self.id_to_token)}", file=sys.stderr)

    def save_vocab(self, output_path):
        """Exports the final vocabulary list to a file."""
        vocab_list = []
        vocab_list.extend(SPECIAL_TOKENS)
        
        # Add merged tokens in order of creation
        for token in self.merge_history:
            if len(vocab_list) >= self.desired_vocab_size:
                break
            vocab_list.append(token)
        
        # Fill remaining slots with base characters/tokens if needed
        if len(vocab_list) < self.desired_vocab_size:
            existing_tokens = set(vocab_list)
            remaining_tokens = [tok for tok in self.id_to_token if tok not in existing_tokens]
            for token in sorted(remaining_tokens):
                if len(vocab_list) >= self.desired_vocab_size:
                    break
                vocab_list.append(token)
        
        final_vocab = vocab_list[:self.desired_vocab_size]

        with open(output_path, "w", encoding="utf-8") as f:
            for token in final_vocab:
                if token in SPECIAL_TOKENS:
                    f.write(token + "\n")
                elif token in self.start_tokens:
                    f.write(token + "\n")
                else:
                    f.write(SUBWORD_PREFIX + token + "\n")

    def build_tokenizer_trie(self):
        """Builds a trie from the learned vocabulary for fast tokenization."""
        self.tokenizer_trie = TokenTrie()
        # Build trie from all possible tokens, not just the final vocab size
        for token in self.id_to_token:
            self.tokenizer_trie.add(token)

    def identify_start_tokens(self, corpus_path):
        """Determines which tokens can appear at the beginning of a word."""
        start_tokens = set()
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                normalized_line = prepare_text(line)
                if not normalized_line:
                    continue
                for word in split_by_whitespace(normalized_line):
                    if not word:
                        continue
                    
                    pos = 0
                    is_first_subword = True
                    while pos < len(word):
                        length, token = self.tokenizer_trie.find_longest_match(word, pos)
                        if length == 0:
                            break # Could not tokenize word fully
                        if is_first_subword:
                            start_tokens.add(token)
                            is_first_subword = False
                        pos += length
        self.start_tokens = start_tokens
    
    def tokenize(self, text_line):
        """Tokenizes a line of text using the learned vocabulary."""
        output_tokens = []
        words = split_by_whitespace(text_line)
        for word in words:
            if not word:
                continue
            
            pos = 0
            word_subwords = []
            is_first = True
            is_tokenization_failed = False
            
            while pos < len(word):
                length, token = self.tokenizer_trie.find_longest_match(word, pos)
                if length == 0:
                    is_tokenization_failed = True
                    break
                
                piece = token if is_first else SUBWORD_PREFIX + token
                word_subwords.append(piece)
                pos += length
                is_first = False
            
            if is_tokenization_failed:
                output_tokens.append("<unk>")
            else:
                output_tokens.extend(word_subwords)
        return output_tokens

    def detokenize(self, tokens):
        """Converts a list of tokens back into a string."""
        reconstructed_words = []
        current_word = ""
        for token in tokens:
            if token == "<unk>":
                if current_word:
                    reconstructed_words.append(current_word)
                    current_word = ""
                reconstructed_words.append(token)
            elif token.startswith(SUBWORD_PREFIX):
                current_word += token[len(SUBWORD_PREFIX):]
            else:
                if current_word:
                    reconstructed_words.append(current_word)
                current_word = token
        if current_word:
            reconstructed_words.append(current_word)
        return " ".join(reconstructed_words)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WordPiece Tokenizer Trainer")
    parser.add_argument("--train", required=True, help="Path to the training text file.")
    parser.add_argument("--input", required=True, help="Path to the file to tokenize.")
    parser.add_argument("--vocab_size", required=True, type=int, help="Desired final vocabulary size.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during training.")
    
    args = parser.parse_args()

    # Define output file names
    rollno = "221093"
    vocab_output_file = f"{rollno}_assignment2_wp_vocab_{args.vocab_size}.txt"
    tokens_output_file = f"{rollno}_assignment2_wp_tokens.txt"
    detokenized_output_file = f"{rollno}_assignment2_wp_detokenized.txt"

    trainer = WordPieceTrainer(vocab_size=args.vocab_size, verbose=args.verbose)
    
    print(" Initializing from training corpus...", file=sys.stderr)
    trainer.load_corpus_and_initialize(args.train)
    
    print(" Starting vocabulary training...", file=sys.stderr)
    trainer.run_training()
    
    print(" Building tokenizer from vocabulary...", file=sys.stderr)
    trainer.build_tokenizer_trie()

    print(" Identifying start-tokens for proper vocab export...", file=sys.stderr)
    trainer.identify_start_tokens(args.train)

    print(f" Saving vocabulary to {vocab_output_file}...", file=sys.stderr)
    trainer.save_vocab(vocab_output_file)

    print(f" Tokenizing input file {args.input}...", file=sys.stderr)
    detokenized_lines = []
    with open(args.input, "r", encoding="utf-8") as f_in, \
         open(tokens_output_file, "w", encoding="utf-8") as f_tok_out:
        for line in f_in:
            original_line = line.strip()
            normalized_line = prepare_text(original_line)
            tokens = trainer.tokenize(normalized_line)
            for t in tokens:
                f_tok_out.write(t + "\n")
            reconstructed_line = trainer.detokenize(tokens)
            detokenized_lines.append(reconstructed_line)

    print(f" Writing detokenized output to {detokenized_output_file}...", file=sys.stderr)
    with open(detokenized_output_file, "w", encoding="utf-8") as f_detok_out:
        f_detok_out.write("\n".join(detokenized_lines))
        
    print("\nProcessing complete.", file=sys.stderr)
