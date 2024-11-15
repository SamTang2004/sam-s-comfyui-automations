import json
import random
from audioop import reverse
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
num_artists = 5
batch_size = 4
min_weight = 0.5
max_weight = 1
use_default_artists = True
enable_weights = True
resting_time = 300 # 5 minutes
runs = 150
allow_duplicated_default_artists = True
# automatically trim the list.

with open("artist_pool.txt", "r") as f:
    pool = list(set([line.strip() for line in f.readlines()]))
    pool.sort()
    f.close()


# automatically trim repeated artists.
file = open("artist_pool.txt", "w")
for artist in pool:
    file.write(artist + "\n")
file.close()

print(pool)


with open("default_artist_headers.json", "r", encoding="utf-8") as f:
    default_artists = json.load(open("default_artist_headers.json", "r", encoding="utf-8"))["list"]
    f.close()

default_artists.sort(key=lambda x : x[1])



# generate sequence in descending order. 1... in logarithmic distribution.

if use_default_artists:
    weight_list = logarithmic_weights(num_artists + len(default_artists), min_weight, max_weight)
else:
    weight_list = logarithmic_weights(num_artists, min_weight, max_weight)

weight_list = list(reversed(weight_list))
print(weight_list)


with open("workflow_api.json", "r", encoding="utf-8") as f:
    next_workflow = json.load(open("workflow_api.json", "r", encoding="utf-8"))
    f.close()




#------------------------------------------------------------------------------------------------------------------------------------

# cycle begin
# every batch 5 runs
# for every batch, rest for 3 minutes before running the next batch
# 100 batches

iBatch = 0
while iBatch < runs:

    # generate sequences
    sequences = generate_sequences(pool, num_artists, batch_size)

    # preprocess sequences into artist strings
    artist_strings = []

    for iseq, seq in enumerate(sequences):\
        # seq is a sequence of artists

        # weight
        if enable_weights:

            next_string_comp = []
            if use_default_artists:
                for next_artist_and_pos in default_artists:
                    seq.insert(next_artist_and_pos[1], next_artist_and_pos[0])

            # remove duplicates or not?
            if not allow_duplicated_default_artists:
                seen = set()
                to_pop = []

                # add dupes to the list
                for i in range(len(seq)):
                    if seq[i] not in seen:
                        seen.add(seq[i])
                    else:
                        to_pop.append(seq[i])

                # if dupe, remove the last occurrence of the duped
                # At max: 1 dupe
                # can only occur after adding in the next artist and pos
                seq = list(reversed(seq))
                for artist in to_pop:
                    seq.remove(artist)
                seq = list(reversed(seq))



            for idx in range(len(seq)):
                next_string_comp.append(f"(artist:{seq[idx]}:{weight_list[idx]})")



            artist_strings.append(", ".join(next_string_comp))


        else:
            artist_strings.append(", ".join([f"(artist:{i})" for i in seq]))

    for next_string in artist_strings:

        next_workflow["34"]["inputs"]["text"] = next_string
        queue_prompt(next_workflow)

    print(artist_strings)
    # sleep for 5 mins
    print(f"Batch {iBatch} has been queued")
    iBatch += 1
    sleep(resting_time)










