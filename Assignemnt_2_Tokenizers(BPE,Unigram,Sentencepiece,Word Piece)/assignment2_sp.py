import argparse
import os
import collections
import time
import heapq
import unicodedata

# --- Constants ---
BYTE_TO_CHAR_MAP = {i: chr(i) for i in range(256)}
CHAR_TO_BYTE_MAP = {v: k for k, v in BYTE_TO_CHAR_MAP.items()}
SPACE_MARKER = " "

# --- Data Structures ---

class Node:
    # A single token in the linked list.
    __slots__ = ['tok_id', 'seq_id', 'pos', 'prev', 'next', 'active']
    def __init__(self, tok_id, seq_id, pos):
        self.tok_id = tok_id
        self.seq_id = seq_id
        self.pos = pos
        self.prev = None
        self.next = None
        self.active = True

class Sequence:
    # Manages a sequence of tokens.
    def __init__(self, token_ids, seq_id):
        self.seq_id = seq_id
        self.head = None
        
        if not token_ids:
            return

        nodes = [Node(tok_id, seq_id, i) for i, tok_id in enumerate(token_ids)]
        for i in range(len(nodes)):
            if i > 0:
                nodes[i].prev = nodes[i-1]
            if i < len(nodes) - 1:
                nodes[i].next = nodes[i+1]
        self.head = nodes[0]
    
    def merge_nodes(self, left, right, new_tok_id):
        # O(1) merge operation.
        if not (left.active and right.active and left.next == right):
            return None
        
        merged = Node(new_tok_id, self.seq_id, left.pos)
        merged.prev = left.prev
        merged.next = right.next
        
        if merged.prev:
            merged.prev.next = merged
        else:
            self.head = merged
        if merged.next:
            merged.next.prev = merged
        
        left.active = False
        right.active = False
        return merged
    
    def get_pairs(self):
        # Get all valid adjacent node pairs.
        pairs = []
        curr = self.head
        while curr and curr.next:
            if curr.active and curr.next.active:
                pairs.append((curr, curr.next))
            curr = curr.next
        return pairs
    
    def get_node_pairs(self, node):
        # Get pairs adjacent to a specific node.
        pairs = []
        if node.prev and node.prev.active:
            pairs.append((node.prev, node))
        if node.next and node.next.active:
            pairs.append((node, node.next))
        return pairs

class PairManager:
    # Tracks pair frequencies and manages the priority queue.
    def __init__(self):
        self.locations = collections.defaultdict(list)
        self.counts = collections.defaultdict(int)
        self.heap = []
        self.timestamp = 0
        self.pair_timestamp = {}
        self.seq_map = {}

    def add_sequence(self, seq):
        self.seq_map[seq.seq_id] = seq
        for left, _ in seq.get_pairs():
            self.add(left)

    def add(self, left_node):
        if not left_node.next: return
        pair = (left_node.tok_id, left_node.next.tok_id)
        self.locations[pair].append(left_node)
        self.counts[pair] += 1
    
    def remove(self, left_node):
        if not left_node.next: return
        pair = (left_node.tok_id, left_node.next.tok_id)
        self.counts[pair] = max(0, self.counts[pair] - 1)
        
    def build_heap(self):
        # Build the priority queue from all initial pairs.
        self.heap = []
        for pair, count in self.counts.items():
            if count > 0:
                self.timestamp += 1
                self.pair_timestamp[pair] = self.timestamp
                heapq.heappush(self.heap, (-count, pair, self.timestamp))
    
    def update_heap(self, pair):
        # Update priority queue with current count for a specific pair.
        if pair in self.counts and self.counts[pair] > 0:
            count = self.counts[pair]
            self.timestamp += 1
            self.pair_timestamp[pair] = self.timestamp
            heapq.heappush(self.heap, (-count, pair, self.timestamp))
    
    def get_best(self):
        # Get highest priority pair with lazy deletion.
        while self.heap:
            neg_count, pair, ts = heapq.heappop(self.heap)
            if ts != self.pair_timestamp.get(pair):
                continue
            
            current_count = self.counts.get(pair, 0)
            if current_count != -neg_count or current_count == 0:
                continue
            
            return pair
        return None
    
    def get_valid_locations(self, pair):
        # Get valid positions for a specific pair.
        if pair not in self.locations:
            return []
        
        valid = []
        for node in self.locations[pair]:
            if (node.active and node.next and node.next.active and
                node.tok_id == pair[0] and node.next.tok_id == pair[1]):
                valid.append(node)
        
        self.locations[pair] = valid
        self.counts[pair] = len(valid)
        return valid

class TokenizerModel:
    # Holds the vocabulary and a trie for fast encoding.
    def __init__(self, merges, id_to_tok):
        self.id_to_tok = id_to_tok
        self.tok_to_id = {t: i for i, t in id_to_tok.items()}
        
        self.trie = {}
        for tok_str, tok_id in self.tok_to_id.items():
            node = self.trie
            for char in tok_str:
                node = node.setdefault(char, {})
            node['#'] = tok_id

# --- Preprocessing ---
def _preprocess(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = text.replace(' ', SPACE_MARKER)
    if not text.startswith(SPACE_MARKER):
        text = SPACE_MARKER + text
    return text

# --- Template Functions ---

def load_training_data(train_path):
    # Load and return raw text for training.
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def train_bpe_tokenizer(text, vocab_size):
    # Learn BPE merges and return vocabulary and tokenizer.
    proc_text = _preprocess(text)
    
    RESERVED = ["<pad>", "<unk>", "<s>", "</s>"]
    tok_to_id = {BYTE_TO_CHAR_MAP[i]: i for i in range(256)}
    
    for t in RESERVED:
        if t not in tok_to_id:
            tok_to_id[t] = len(tok_to_id)
    
    id_to_tok = {i: t for t, i in tok_to_id.items()}
    
    manager = PairManager()
    
    byte_toks = [tok_to_id[BYTE_TO_CHAR_MAP[b]] for b in proc_text.encode('utf-8')]
    
    seq = Sequence(byte_toks, 0)
    manager.add_sequence(seq)
    
    manager.build_heap()
    
    merge_rules = []
    num_merges = vocab_size - len(tok_to_id)
    
    for _ in range(num_merges):
        best_pair = manager.get_best()
        if not best_pair:
            break
            
        new_id = len(tok_to_id)
        p1, p2 = best_pair
        new_tok_str = id_to_tok[p1] + id_to_tok[p2]
        
        tok_to_id[new_tok_str] = new_id
        id_to_tok[new_id] = new_tok_str
        merge_rules.append(best_pair)
        
        affected = set()
        
        for left_node in manager.get_valid_locations(best_pair):
            right_node = left_node.next
            if not right_node: continue
            
            if left_node.prev:
                manager.remove(left_node.prev)
                affected.add((left_node.prev.tok_id, left_node.tok_id))
            manager.remove(left_node)
            if right_node.next:
                manager.remove(right_node)
                affected.add((right_node.tok_id, right_node.next.tok_id))
            
            seq = manager.seq_map[left_node.seq_id]
            merged = seq.merge_nodes(left_node, right_node, new_id)
            
            if merged:
                for new_left, _ in seq.get_node_pairs(merged):
                    manager.add(new_left)
                    affected.add((new_left.tok_id, new_left.next.tok_id))

        for p in affected:
            manager.update_heap(p)
            
    vocab = []
    vocab.extend(RESERVED)
    
    byte_toks = [BYTE_TO_CHAR_MAP[i] for i in range(256) if BYTE_TO_CHAR_MAP[i] not in RESERVED]
    vocab.extend(byte_toks)
    
    for p1, p2 in merge_rules:
        merged_str = id_to_tok[p1] + id_to_tok[p2]
        if merged_str in tok_to_id:
            vocab.append(merged_str)
            
    tokenizer_model = TokenizerModel(merge_rules, id_to_tok)
    return vocab, tokenizer_model

def tokenize(text, tokenizer):
    # Tokenize input text using trained tokenizer model.
    proc_text = _preprocess(text)
    proc_chars = [BYTE_TO_CHAR_MAP[b] for b in proc_text.encode('utf-8')]
    
    output = []
    i = 0
    while i < len(proc_chars):
        best_len = 0
        best_id = None
        
        curr_trie = tokenizer.trie
        
        for j in range(i, len(proc_chars)):
            char = proc_chars[j]
            if char in curr_trie:
                curr_trie = curr_trie[char]
                if '#' in curr_trie:
                    best_len = j - i + 1
                    best_id = curr_trie['#']
            else:
                break
        
        if best_len == 0:
            output.append(proc_chars[i])
            i += 1
        else:
            output.append(tokenizer.id_to_tok[best_id])
            i += best_len
    return output

def detokenize(tokens, tokenizer):
    # Detokenize tokens back to original text.
    token_str = "".join(tokens)
    text_bytes = bytearray()
    
    for char in token_str:
        if char in CHAR_TO_BYTE_MAP:
            text_bytes.append(CHAR_TO_BYTE_MAP[char])

    text = text_bytes.decode('utf-8', 'replace')
    text = text.replace(SPACE_MARKER, ' ')
    return text.strip()

def save_vocab(vocab, rollno, vocab_size):
    # Save vocabulary file in required format.
    fname = f"{rollno}_assignment2_spm_vocab_{len(vocab)}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    print(f"Vocabulary saved to {fname}")

def save_tokens(tokens, rollno):
    # Save tokens to a file.
    fname = f"{rollno}_assignment2_spm_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))
    print(f"Tokens saved to {fname}")

def save_detokenized(text, rollno):
    # Save detokenized text to a file.
    fname = f"{rollno}_assignment2_spm_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Detokenized text saved to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno= "221093"

    train_text = load_training_data(args.train)
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)