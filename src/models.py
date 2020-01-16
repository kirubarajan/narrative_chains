import math

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