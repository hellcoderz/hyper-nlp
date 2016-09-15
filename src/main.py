
import random

def get_rand_vector(total_bits=1000, active_bits_prcentage=2):
    """return random list of activated cells"""
    active_bits_num = int(total_bits*active_bits_prcentage*1.0/100)
    active_cells = []
    for _ in range(active_bits_num):
        r = random.randint(0, 999)
        active_cells.append(r)
    return active_cells

def build_vocab(train_samples):
    """build a vocab of word => active cell index list"""
    vocab = {}
    for tup in train_samples:
        words = tup[0].split()
        for word in words:
            if not word in vocab:
                vocab[word] = get_rand_vector()
    return vocab

if __name__ == "__main__":
    data = [
        ("watch harry potter", "TV"),
        ("show mw action movies", "TV")
    ]
    vocab = build_vocab(data)