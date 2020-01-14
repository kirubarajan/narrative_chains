import math
import neuralcoref
import spacy
from collections import defaultdict
from spacy.symbols import nsubj, nsubjpass, VERB

# set constants
INPUT_FILE = "data/input1.txt"
OUTPUT_FILE = "export.txt"
MAX_LENGTH = 1_000_000
CHUNK_LENGTH = 100_000

##### TODO: chunk and support multiple files
# clean data
# loader = build_loader(INPUT_FILE)
# text = loader.get_text()[:MAX_LENGTH] 

with open(INPUT_FILE) as f:
    text = " ".join(f.readlines()[13:-7])

# identify events
subjects = defaultdict(lambda: defaultdict(int))
objects = defaultdict(lambda: defaultdict(int))
total = 0

# chunking text and parsing
for i in range(0, MAX_LENGTH, CHUNK_LENGTH):
    chunk = text[i:i + CHUNK_LENGTH]

    # resolve entities and gramatically parse 
    nlp = spacy.load("en")
    neuralcoref.add_to_pipe(nlp)
    entitied_text = nlp(chunk.lower())._.coref_resolved
    corpus = nlp(entitied_text)

    for token in corpus:
        if token.pos == VERB:
            for argument in token.children:
                if argument.dep_ in {"nsubj", "nsubjpass"}:
                    subjects[token.lemma_][argument.text] += 1
                    total += 1
                elif argument.dep_ in {"dobj", "iobj", "pobj", "obj"}:
                    objects[token.lemma_][argument.text] += 1
                    total += 1
    
    print("completed chunk ", int(i / CHUNK_LENGTH))

verbs = set(subjects.keys()) | set(objects.keys())
print("# of verbs: ", len(verbs))

# create coreference matrix
coreference = defaultdict(lambda: defaultdict(int))
total_coreference = 0

for verb1 in verbs:
    for verb2 in verbs:
        verb1_subjects = set(subjects[verb1].keys())
        for argument in subjects[verb2]:
            if argument in verb1_subjects:
                coreference[verb1][verb2] += 1
                total_coreference += 1

        verb1_objects = set(objects[verb1].keys())
        for argument in objects[verb2]:
            if argument in verb1_objects:
                coreference[verb1][verb2] += 1
                total_coreference += 1

# write events to output file
with open(OUTPUT_FILE, "w") as file:
    for verb in verbs:
        for subj in subjects[verb]:
            file.write("\n")
            file.write(verb + " " + subj + " " + "subj")

        for obj in objects[verb]:
            file.write("\n")
            file.write(verb + " " + obj + " " + "obj")

# marginal probability of event: P(e)
def marginal(event):
    verb, dependency, dep_type = event
    frequency = sum([subjects[verb][x] for x in subjects[verb]]) + sum([objects[verb][x] for x in objects[verb]])
    return frequency / total

# joint probability of two events
def joint(event1, event2):
    verb1, verb2 = event1[0], event2[0]
    return (coreference[verb1][verb2] + coreference[verb2][verb1]) / total_coreference

# pointwise mutual information approximation of two events
def pmi(event1, event2):
    numerator = joint(event1, event2)
    denominator = math.exp(math.log(marginal(event1)) + math.log(marginal(event2)))
    return 0.0 if numerator == 0 else math.log(numerator / denominator)

print("total coreference count: ", total_coreference)

# testing random things
event = ("say", "somebody", "subj")
print("\nevent: ", event)
print("marginal: ", marginal(event))

event = ("think", "i", "subj")
print("\nevent: ", event)
print("marginal: ", marginal(event))

print("coreference count of think/know: ", coreference["think"]["know"])
print("coreference count of count/surpass: ", coreference["count"]["surpass"])

"""
event1 = ("think", "i", "subj")
event2 = ("know", "i", "subj")
print("\nevents: ", event1, event2)
print("joint probability of events: ", joint(event1, event2))
print("pmi of events:", pmi(event1, event2))

event2 = ("skyrocket", "i", "subj")
print("\nevents: ", event1, event2)
print("joint probability of events: ", joint(event1, event2))
print("pmi of events:", pmi(event1, event2))
"""

def predict(chain):
    scores = dict()
    for verb in verbs:
        score = 0
        for event in chain:
            score += pmi(event, (verb, None, None))
        scores[verb] = score

    cleaned_scores = dict()
    chain_verbs = set()
    for event in chain:
        chain_verbs.add(event)

    for candidate in scores:
        if candidate not in chain_verbs:
            cleaned_scores[candidate] = scores[candidate]
    
    ranked_scores = sorted(list(cleaned_scores.items()), key=lambda x: x[1], reverse=True)
    return ranked_scores

chain = [("be", "Townley", "subj"), ("say", "Townley", "subj")]
prediction = predict(chain)[0]
print("\nchain: ", chain)
print("prediction (and score): ", prediction)