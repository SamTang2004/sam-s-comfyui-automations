import random
from difflib import SequenceMatcher


def initialize_first_sequence(pool, N):
    return random.sample(pool, N)


def compute_similarity(seq1, seq2):

    return SequenceMatcher(None, seq1, seq2).ratio()


def select_next_sequence(pool, current_sequences, N, threshold=0.1):
    best_sequence = None
    best_score = -float('inf')
    # bug: Can have zero sequence.

    while not best_sequence:
        for _ in range(100):  # Generate 100 candidate sequences, could tune this
            candidate = random.sample(pool, N)
            min_similarity = min(compute_similarity(candidate, seq) for seq in current_sequences)

            # Prioritize sequences with fewer overlapping artists (lower similarity score)
            if min_similarity >= threshold:
                score = min_similarity
                if score > best_score:
                    best_score = score
                    best_sequence = candidate

    return best_sequence


def generate_sequences(pool, N, num_sequences):
    sequences = []
    sequences.append(initialize_first_sequence(pool, N))

    for _ in range(1, num_sequences):
        next_sequence = select_next_sequence(pool, sequences, N)
        sequences.append(next_sequence)

    return sequences


# Set parameters
pool = [chr(97 + i) for i in range(20)]  # ['a', 'b', 'c', ..., 't']
N = 10
num_sequences = 5

# Generate sequences
sequences = generate_sequences(pool, N, num_sequences)

artist_strings = []
for iseq, seq in enumerate(sequences):

    artist_strings.append(", ".join([f"(artist:{i})" for i in seq]))


print(artist_strings)

