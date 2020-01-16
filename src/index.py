import sys
import pickle
import math
import neuralcoref
import spacy
from collections import defaultdict
from spacy.symbols import nsubj, nsubjpass, VERB

if len(sys.argv) == 2 and sys.argv[1] == "--train":
    print("\nRunning Narrative Chain Indexing")

    # set constants
    INPUT_FILE = "data/input1.txt"
    OUTPUT_FILE = "export.txt"
    MAX_LENGTH = 50_000
    CHUNK_LENGTH = 10_000

    # read file and clean input
    with open(INPUT_FILE) as f:
        text = " ".join(f.readlines()[13:-7])

    # identify events
    ordered = list()
    subjects = defaultdict(lambda: defaultdict(int))
    objects = defaultdict(lambda: defaultdict(int))
    total = 0

    # chunking text and parsing
    for i in range(0, MAX_LENGTH, CHUNK_LENGTH):
        chunk = text[i:i + CHUNK_LENGTH]
        print("chunk ", int(i / CHUNK_LENGTH))

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
                        ordered.append((token.lemma_, argument.text, argument.dep_))
                    elif argument.dep_ in {"dobj", "iobj", "pobj", "obj"}:
                        objects[token.lemma_][argument.text] += 1
                        total += 1
                        ordered.append((token.lemma_, argument.text, argument.dep_))

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

    # write events to output file
    with open(OUTPUT_FILE, "w") as file:
        for event in ordered:
            file.write("\n" + str(event))

    # TODO: pickle subjects, objects, coreference, total
    class Model: pass
    model = Model()
    model.subjects, model.objects, model.coreference = dict(subjects), dict(objects), dict(coreference)
    model.total, model.total_coreference = total, total_coreference

    print("\nDumping Model")
    with open("model.pickle", "wb") as file:
        pickle.dump(model, file)
    print("successfully saved to model.pickle")

else:
    with open("model.pickle", "rb") as file:
        class Model: pass
        model = pickle.load(file)
        total, total_coreference = model.total, model.total_coreference
        subjects, objects, coreference = defaultdict(lambda: defaultdict(int), model.subjects), defaultdict(lambda: defaultdict(int), model.objects), defaultdict(lambda: defaultdict(int), model.coreference)
        verbs = set(subjects.keys()) | set(objects.keys())

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
    marginal1, marginal2 = marginal(event1), marginal(event2)
    if marginal1 == 0 or marginal2 == 0 or numerator == 0: return 0.0

    denominator = math.exp(math.log(marginal1) + math.log(marginal2))
    return math.log(numerator / denominator)

# chain prediction
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

# testing narrative cloze
testing_pairs = [
    ([("receive", "clients", "nsubj"), ("download", "clients", "dobj")], ("make", "clients", "nsubj")),
    ([("fled", "gelman", "nsubj"), ("found", "gelman", "nsubj")], ("take", "gelman", "nsubj")),
    ([("am", "i", "nsubj"), ("did", "i", "nsubj"), ("think", "i", "nsubj")], ('believe', "i", "nsubj")),
    ([("bought", "team", "dobj"), ("included", "team", "dobj")], ("take", "team", "nsubj")),
    ([("heard", "parents", "nsubj"), ("talking", "parents", "nsubj")], ("choose", "parents", "nsubj")),
    ([("buy", 'stock', 'dobj'), ("lend", 'money', 'dobj')], ("struggle", 'edison', 'nsubj')),
    ([("advocated", 'league', 'nsubj'), ("fought", 'league', 'nsubj')], ("withdraw", 'league', 'nsubj')),
    ([("was", 'cranston', 'nsubj'), ("spent", 'cranston', 'nsubj'), ("fight", 'cranston', 'nsubj')], ("raise", 'cranston', 'nsubj')),
    ([("have", 'administration', 'nsubj'), ("convinced", 'administration', 'nsubj'), ("look", 'administration', 'nsubj')], ("push", 'administration', 'nsubj')),
    ([('hug', 'father', 'dobj'), ('tell', 'father', 'dobj')], ('love', 'father', 'dobj')),
    ([('be', 'i', 'nsubj'), ('get', 'i', 'nsubj'), ('have', 'i', 'nsubj')], ('call', 'i', 'nsubj'))
]

def get_position(predictions, correct):
    for i in range(len(predictions)):
        if predictions[i][0] == correct[0]:
            return i + 1
    return len(predictions)

print("\nEvaluating Narrative Cloze Positions: ")
positions = list()
for chain, correct in testing_pairs:
    predictions = predict(chain)
    position = get_position(predictions, correct)
    positions.append(position)
    print("position: ", position)

# computing averages
average = sum(positions) / len(positions)
print("\naverage position: ", average)

adjusted_average = sum([x for x in positions if x != len(verbs)]) / len([x for x in positions if x != len(verbs)])
print("adjusted average position: ", adjusted_average)