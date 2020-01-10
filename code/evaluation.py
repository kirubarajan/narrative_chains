"""Narrative Cloze Evaluation Module"""
import random
from models import predict_events
from pymagnitude import Magnitude

# generates a prediction/answer pair given a chain
# an already cleaned chain can be given an index of -1
def predict_blank(chain, document, index=None):
    if index == None: index = random.randrange(len(chain))

    cleaned_chain = list()
    for i in range(len(chain)): 
        if i != index:
            cleaned_chain.append(chain[i]) 

    predictions = predict_events(cleaned_chain, document, n=len(document.events))

    return predictions, chain[index]

# helper to return a model's prediction position
def get_position(predictions, correct, embedding=False):
    for i in range(len(predictions)):
        if predictions[i] == correct:
            return i + 1
    return len(predictions)

# assigns an average position to a test set of chains
def score_predictions(test_pairs, document, embedding=False):
    position_score = 0
    for chain, correct in test_pairs:
        predictions, _ = predict_blank(chain, document)
        position_score += get_position(predictions, correct, embedding)
    return position_score / len(test_pairs)