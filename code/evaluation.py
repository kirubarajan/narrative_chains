"""Narrative Cloze Evaluation Module"""
import copy
import random
from models import Event, predict_events
from pymagnitude import Magnitude

# generates a prediction/answer pair given a chain
# an already cleaned chain can be given an index of -1
def predict_blank(chain, document, index=None, embedding=False):
    if index == None: index = random.randrange(len(chain))

    cleaned_chain = list()
    for i in range(len(chain)): 
        if i != index:
            cleaned_chain.append(chain[i]) 

    predictions = predict_events(cleaned_chain, document, n=len(document.events), embedding=embedding)

    return predictions, chain[index]

# helper to return a model's prediction position
def get_position(predictions, correct, embedding=False):
    for i in range(len(predictions)):
        if predictions[i].verb == correct.verb:
            return i + 1
    return len(predictions)

# assigns an average position to a test set of chains
def score_predictions(test_pairs, document, embedding=False):
    position_score = 0
    for chain, correct in test_pairs:
        predictions, _ = predict_blank(chain, document, embedding=embedding)
        position = get_position(predictions, correct, embedding)
        print("position: ", position)
        position_score += position
    return position_score / len(test_pairs)

"""NYT Annotated Chains"""
# chains are hand-annotated from the input corpus, not the output of identified events (manually lemmatizing)
chain0 = [Event("They", "predict", "uprising", "dobj"), Event("It", "cost", "100,000", "dobj"), Event("Police", "tip", "Taliban", "dobj")]
chain1 = [Event("Michiganians", "contribute", "money", "dobj"), Event("Citizens", "give", "50k", "dobj")]
chain2 = [Event("Kevin", "started", "game", "dobj"), Event("Kevin", "played", "game", "dobj")]
chain3 = [Event("He", "jumped", "up", "dobj"), Event("He", "fell", "down", "dobj")]
chain4 = [Event("Somebody", "joined", "navy", "dobj"), Event("Somebody", "served", "navy", "dobj")]
chain5 = [Event("Markers", "tell", "them", "dobj"), Event("He", "play", "man", "dobj"), Event("He", "think", "whatever", "dobj")]
chain6 = [Event("Ballmer", "use", "Show", "dobj"), Event("Phones", "take", "capabilities", "dobj"), Event("Ballmer", "announces", "services", "dobj"), Event("Microsoft", "ships", "licenses", "dobj")]
chain7 = [Event("People", "buy", "cars", "dobj"), Event("I", "like", "it", "dobj"), Event("He", "channel", "Nixon", "dobj"), Event("One", "aquire", "pawn", "dobj")]
chain8 = [Event("Numbers", "dropped", "50,000", "dobj"), Event("Stores", "hire", "more", "people"), Event("Development", "was", "influx", "dobj")]
chain9 = [Event("He", "introduced", "women", "dobj"), Event("He", "dating", "many", "dobj")]
chain10 = [Event("They", "make", "music", "dobj"), Event("Bowlen", "want", "franchise", "dobj"), Event("He", "discuss", "dissmissal", "dobj")]
chain11 = [Event("People", "ask", "me", "dobj"), Event("Players", "compliment", "coaches", "dobj"), Event("I", "say", "thanks", "dobj"), Event("He", "characterize", "whom", "dobj")]
chain12 = [Event("Law", "give", "latitude", "dobj"), Event("They", "spend", "donations", "dobj")]
chain13 = [Event("Inauguration", "cost", "million", "dobj"), Event("It", "draw", "crowd", "dobj"), Event("Conybeare", "give", "25,000", "dobj"), Event("Administration", "help", "economy", "dobj")]
chain14 = [Event("NHTSA", "open", "investigation", "dobj"), Event("Martin", "identify", "reason", "dobj"), Event("NHTSA", "report", "complaints", "dobj")]

output0 = Event("Police", "demand", "4,000", "dobj")
output1 = Event("Democrats", "build", "majority", "dobj")
output2 = Event("Kevin", "won", "game", "dobj")
output3 = Event("He", "hurt", "himself", "dobj")
output4 = Event("Somebody", "quit", "navy", "dobj")
output5 = Event("He", "dominate", "them", "dobj")
output6 = Event("Carrier", "offer", "phones", "dobj")
output7 = Event("Austin", "pay", "130,000,000", "dobj")
output8 = Event("Sales", "surpass", "previous", "dobj")
output9 = Event("He", "married", "favorite")
output10 = Event("Owner", "call", "conference", "dobj")
output11 = Event("He", "talk", "me", "dobj")
output12 = Event("They", "buy", "things", "dobj")
output13 = Event("Dries", "contribute", "250", "dobj")
output14 = Event("NHTSA", "report", "complaints", "dobj")

nyt_test_pairs = [(chain0, output0), (chain1, output1), (chain2, output2), (chain3, output3), (chain4, output4), (chain5, output5), (chain6, output6), (chain7, output7), (chain8, output8), (chain9, output9), (chain10, output10), (chain11, output11), (chain12, output12), (chain13, output13), (chain14, output14)]

"""Common Sense Reasoning Annotated Chains"""
c_chain0 = [Event("I", "start", "game", "dobj"), Event("I", "play", "game", "dobj")]
c_output0 = Event("I", "win", "game", "dobj")

common_sense_test_pairs = [(c_chain0, c_output0)]