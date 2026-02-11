import argparse
import os
import collections
import time
import heapq

# Constants
BYTE_TO_CHAR_MAP = {i: chr(i) for i in range(256)}
CHAR_TO_BYTE_MAP = {v: k for k, v in BYTE_TO_CHAR_MAP.items()}

# Data Structures defintions for Linked List and Heaps

class Node:
    def __init__(self, tok_id, seq_id, pos):
        self.tok_id = tok_id
        self.seq_id = seq_id # Reference to parent's word id
        self.pos = pos
        self.prev = None
        self.next = None 
        self.active = True # For handling deletions correctly
        self.parent = None # Reference to parent Word

class Word:
    def __init__(self, token_ids, word_id):
        # making seqeunce of nodes i.e linked list from list of token ids
        self.word_id = word_id
        self.head = None
        
        nodes = []
        for i, tok_id in enumerate(token_ids):
            node = Node(tok_id, word_id, i)
            node.parent = self
            nodes.append(node)
            if i > 0:
                nodes[i-1].next = node
                node.prev = nodes[i-1]
        
        if nodes:
            self.head = nodes[0]

    def merge_nodes(self, left, right, new_tok_id):
        # merge operation
        if not (left.active and right.active and left.next == right):
            return None
        
        merged = Node(new_tok_id, self.word_id, left.pos)
        merged.parent = self
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
        # getting all valid adjacent pairs
        pairs = []
        curr = self.head
        while curr and curr.next:
            if curr.active and curr.next.active:
                pairs.append((curr, curr.next))
            curr = curr.next
        return pairs

class PairManager:
    # tracking pairs and frequency alogwith correct merge and updations
    # we use heap here
    def __init__(self):
        # locations maintains all the location of a pair of token ids(e.g. (72, 69)) in a list
        self.locations = collections.defaultdict(list)
        self.counts = collections.defaultdict(int)
        self.heap = []
        self.timestamp = 0
        self.pair_timestamp = {}

    def build_heap(self):
        self.heap = []
        for pair, count in self.counts.items():
            if count > 0:
                self.timestamp += 1
                self.pair_timestamp[pair] = self.timestamp
                heapq.heappush(self.heap, (-count, pair, self.timestamp))
    
    def add(self, left_node, right_node):
        #maintaining the locations and counts of pairs
        pair = (left_node.tok_id, right_node.tok_id)
        self.locations[pair].append(left_node)
        self.counts[pair] += 1
    

    def update_heap(self, pair):
        if pair in self.counts and self.counts[pair] > 0:
            count = self.counts[pair]
            self.timestamp += 1
            self.pair_timestamp[pair] = self.timestamp
            heapq.heappush(self.heap, (-count, pair, self.timestamp))

    def get_best(self):
        # VVI function: get the best pair from heap for merging
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
        # this is to get a list of valid locations for a pair
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
    #this stores the learned merge rules and vocabulary mapping 
    def __init__(self, merges, id_to_tok):
        self.merges = merges
        self.id_to_tok = id_to_tok

def load_training_data(train_path):
    """Load and return raw text for training."""
    with open(train_path, "r", encoding="utf-8") as f:
        return f.read()

def train_bpe_tokenizer(text, vocab_size):
    """Learn BPE merges and return vocabulary."""
    RESERVED = ["<pad>", "<unk>", "<s>", "</s>"]
    EOW = "</w>"
    
    # create intial vocabulary
    tok_to_id = CHAR_TO_BYTE_MAP.copy()
    for t in RESERVED + [EOW]:
        if t not in tok_to_id:
            tok_to_id[t] = len(tok_to_id)
    
    id_to_tok = {i: t for t, i in tok_to_id.items()}
    
    #getting word frequencies
    word_freqs = collections.Counter(text.split())
    
    manager = PairManager()
    unique_words = {}
    #go through each unique word and create linked list
    # of its tokens and put their pairs in the heap
    for i, word_str in enumerate(word_freqs):
        tok_ids = [tok_to_id[BYTE_TO_CHAR_MAP[b]] for b in word_str.encode('utf-8')]
        tok_ids.append(tok_to_id[EOW])
        
        word_seq = Word(tok_ids, i)
        unique_words[word_str] = (word_seq, i)
        
        for left, right in word_seq.get_pairs():
            manager.add(left, right)

    for word_str, freq in word_freqs.items():
        if freq > 1:
            word_seq, _ = unique_words[word_str]
            for left, right in word_seq.get_pairs():
                pair = (left.tok_id, right.tok_id)
                manager.counts[pair] += freq - 1

    manager.build_heap()
    
    merge_rules = []
    num_merges = vocab_size - len(tok_to_id)
    word_id_to_freq = {i: freq for i, (word, freq) in enumerate(word_freqs.items())}

    for _ in range(num_merges):
        # VVI part: get the most frequent pair from our manager.
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
        #For each occurrence of the best pair
        #  merge them and update pairs
        for left_node in manager.get_valid_locations(best_pair):
            right_node = left_node.next
            if not right_node: continue
            
            word_obj = left_node.parent
            word_freq = word_id_to_freq.get(word_obj.word_id, 1)

            if left_node.prev and left_node.prev.active:
                old = (left_node.prev.tok_id, left_node.tok_id)
                affected.add(old)
                manager.counts[old] -= word_freq
            
            if right_node.next and right_node.next.active:
                old = (right_node.tok_id, right_node.next.tok_id)
                affected.add(old)
                manager.counts[old] -= word_freq
                
            merged = word_obj.merge_nodes(left_node, right_node, new_id)
            if merged:
                if merged.prev and merged.prev.active:
                    new = (merged.prev.tok_id, merged.tok_id)
                    affected.add(new)
                    manager.add(merged.prev, merged)
                    manager.counts[new] += word_freq - 1
                
                if merged.next and merged.next.active:
                    new = (merged.tok_id, merged.next.tok_id)
                    affected.add(new)
                    manager.add(merged, merged.next)
                    manager.counts[new] += word_freq - 1
                    
        for p in affected:
            manager.update_heap(p)         
    
    vocab = []
    specials = RESERVED + [EOW]
    vocab.extend(specials)
    
    byte_toks = [BYTE_TO_CHAR_MAP[i] for i in range(256) if BYTE_TO_CHAR_MAP[i] not in specials]
    vocab.extend(byte_toks)
    
    #adding merged tokens to vocabulary
    for p1, p2 in merge_rules:
        merged_str = id_to_tok[p1] + id_to_tok[p2]
        vocab.append(merged_str)
        
    tokenizer_model = TokenizerModel(merge_rules, id_to_tok)
    return vocab, tokenizer_model


def save_vocab(vocab, rollno, vocab_size):
    """Save vocabulary file in required format."""
    fname = f"{rollno}_assignment2_bpe_vocab_{len(vocab)}.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
    print(f"Vocabulary saved to {fname}")

def tokenize(text, tokenizer):
    """Tokenize input text using trained BPE model."""
    EOW = "</w>"
    all_tokens = []
    
    for word in text.split():
        word_tokens = [BYTE_TO_CHAR_MAP[b] for b in word.encode('utf-8')] + [EOW]
        # applying merges in order
        for pair in tokenizer.merges:
            p1_str = tokenizer.id_to_tok[pair[0]]
            p2_str = tokenizer.id_to_tok[pair[1]]
            new_list = []
            i = 0
            while i < len(word_tokens):
                if i < len(word_tokens) - 1 and word_tokens[i] == p1_str and word_tokens[i+1] == p2_str:
                    new_list.append(p1_str + p2_str)
                    i += 2
                else:
                    new_list.append(word_tokens[i])
                    i += 1
            word_tokens = new_list
        all_tokens.extend(word_tokens)
    return all_tokens

def detokenize(tokens, tokenizer):
    """Detokenize tokens back to original text."""
    EOW = "</w>"
    text_bytes = bytearray()
    
    for token in tokens:
        if token.endswith(EOW):
            prefix = token[:-len(EOW)]
            if prefix:
                for char in prefix:
                    if char in CHAR_TO_BYTE_MAP:
                        text_bytes.append(CHAR_TO_BYTE_MAP[char])
            text_bytes.extend(b' ')
        elif token not in ["<pad>", "<unk>", "<s>", "</s>"]:
            for char in token:
                if char in CHAR_TO_BYTE_MAP:
                    text_bytes.append(CHAR_TO_BYTE_MAP[char])
    return text_bytes.strip().decode('utf-8', 'replace')


def save_tokens(tokens, rollno):
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))
    print(f"Tokens saved to {fname}")

def save_detokenized(text, rollno):
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Detokenized text saved to {fname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    # Replace with your actual roll number
    rollno = "221093"

    # Training
    train_text = load_training_data(args.train)
    vocab, tokenizer = train_bpe_tokenizer(train_text, args.vocab_size)
    save_vocab(vocab, rollno, args.vocab_size)

    # Tokenization
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    tokens = tokenize(sample_text, tokenizer)
    save_tokens(tokens, rollno)

    # Detokenization
    detok_text = detokenize(tokens, tokenizer)
    save_detokenized(detok_text, rollno)