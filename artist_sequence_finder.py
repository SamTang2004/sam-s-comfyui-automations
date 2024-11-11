import json
import random
from difflib import SequenceMatcher
from time import sleep
import numpy as np
from urllib import request, parse


def initialize_first_sequence(pool, N):
    return random.sample(pool, N)


def compute_similarity(seq1, seq2):

    return SequenceMatcher(None, seq1, seq2).ratio()


def logarithmic_weights(num_artists, min_weight=0.5, max_weight=1.0):
    # Generate artist indices from 1 to num_artists
    indices = np.arange(1, num_artists + 1)

    # Apply log scaling; using log base 10 for this example
    log_values = np.log(indices + 1)  # Adding 1 to avoid log(1)=0 for first artist

    # Normalize the log values to fit in the range [min_weight, max_weight]
    min_log, max_log = log_values.min(), log_values.max()
    weights = min_weight + (log_values - min_log) * (max_weight - min_weight) / (max_log - min_log)

    return weights


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

def queue_prompt(prompt:dict):
    p = {"prompt": prompt}

    data = json.dumps(p).encode('utf-8')
    req =  request.Request("http://127.0.0.1:8188/prompt", data=data)
    request.urlopen(req)



# Params for testing.
#------------------------------------------------------------------------------------------------------------------------------------
pool = [line.strip() for line in open("artist_pool.txt", "r").readlines()]
num_artists = 7
batch_size = 5

# generate sequence in descending order. 1... in logarithmic distribution.
weight_list = logarithmic_weights(num_artists)
weight_list = list(reversed(weight_list))

enable_weights = True
next_workflow = json.load(open("workflow_api.json", "r", encoding="utf-8"))

#------------------------------------------------------------------------------------------------------------------------------------

# cycle begin
# every batch 5 runs
# for every batch, rest for 3 minutes before running the next batch
# 100 batches

iBatch = 0
while iBatch < 100:

    # generate sequences
    sequences = generate_sequences(pool, num_artists, batch_size)

    # preprocess sequences into artist strings
    artist_strings = []

    for iseq, seq in enumerate(sequences):

        # weight
        if enable_weights:
            artist_strings.append(", ".join([f"(artist:{i}:{weight_list[iseq]})" for i in seq]))
        else:
            artist_strings.append(", ".join([f"(artist:{i})" for i in seq]))

    for next_string in artist_strings:

        next_workflow["34"]["inputs"]["text"] = next_string
        queue_prompt(next_workflow)

    # sleep for 5 mins
    print(f"Batch {iBatch} has been queued")
    sleep(300)









