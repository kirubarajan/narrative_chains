import neuralcoref
import spacy
from collections import defaultdict

# identify events
ordered = list()
subjects = defaultdict(lambda: defaultdict(int))
objects = defaultdict(lambda: defaultdict(int))
total = 0

# chunking text and parsing
spacy.prefer_gpu()

for i in range(0, MAX_LENGTH, CHUNK_LENGTH):
    chunk = text[i:i + CHUNK_LENGTH]
    print("\nchunk ", int(i / CHUNK_LENGTH))

    # resolve entities and gramatically parse 
    print("parsing chunk")
    nlp = spacy.load("en")
    neuralcoref.add_to_pipe(nlp)
    corpus = nlp(chunk)

    print("mining events")
    for token in corpus:
        if token.pos == spacy.symbols.VERB:
            for argument in token.children:
                # resolve argument coreference entity
                if argument._.in_coref: esolved = argument._.coref_clusters[0].main.text
                else: resolved = argument.text

                if argument.dep_ in {"nsubj", "nsubjpass"}:
                    subjects[token.lemma_.lower()][argument.text.lower()] += 1
                    ordered.append((token.lemma_, resolved.lower(), argument.dep_))
                    total += 1
                elif argument.dep_ in {"dobj", "iobj", "pobj", "obj"}:
                    objects[token.lemma_.lower()][argument.text.lower()] += 1
                    ordered.append((token.lemma_, resolved.lower(), argument.dep_))
                    total += 1

verbs = set(subjects.keys()) | set(objects.keys())
print("total verb count: ", len(verbs))

# create coreference matrix
print("\nComputing Coreference Matrix")

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

print("total coreference count: ", total_coreference)