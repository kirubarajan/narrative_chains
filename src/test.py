"""Script for testing document parsing and narrative chain creation"""

from data import build_loader, export_events
from parse import parse_document
from models import Event, event_prob, joint_event_prob, pmi_approx, predict_events
from evaluation import predict_blank, score_predictions, get_position, nyt_test_pairs, common_sense_test_pairs

"""Data Pre-Processing"""
file_path = "data/agiga/export.txt"
loader = build_loader(file_path)
# loader.sanity_check()

EMBEDDING = True
COREF = True
EXAMPLE = "Kevin joined the army. He served the army. He then oversaw the army. Finally, he resigned from the navy." 
MAX_LENGTH = 10000
text = loader.get_text()[:MAX_LENGTH]
# text = EXAMPLE

document = parse_document(text, coref=COREF, lemma=True)
export_events(document, "events_export.txt")
events = list(document.events.items())

print("\nParsing Statistics: ")
print("events: ", len(events))
print("verbs: ", len(document.verbs))
print("dependencies: ", len(document.dependencies))
print("dependency types: ", len(document.dependency_types))

def test_prior():
    print("\nTesting Prior Probability of Each Event")
    for event in document.left_events:
        print(event, event_prob(event, document))

    for event in document.right_events:
        print(event, event_prob(event, document))
    
def test_joint(): 
    print("\nTesting Joint Probability of Events")
    
    for i in range(len(document.left_events) - 1):
        event1, event2 = document.left_events[i], document.left_events[i + 1]
        print(event1, event2, joint_event_prob(event1, event2, document))

def test_pmi():
    print("\ntesting pointwise mutual information approximation of events")
    for i in range(len(document.left_events) - 1):
        event1, event2 = document.left_events[i], document.left_events[i + 1]
        print(event1, event2, pmi_approx(event1, event2, document))

def test_prediction():
    print("\nTesting Event Prediction")
    event1, event2 = Event("join", "somebody", "nsubj"), Event("serve", "somebody", "nsubj")
    chain = [event1, event2]
    # print(event_prob(event1, document))
    # print(event_prob(event2, document))
    # print(joint_event_prob(event1, events[1][0], document))
    
    print("\nCurrent Chain: ")
    for event in chain: 
        print(event.verb, event.dependency_type)
    
    print("\nRanked Predictions and Scores:")
    ranked_predictions = predict_events(chain, document, n=50, embedding=EMBEDDING, include_ranks=True)
    for prediction, score in ranked_predictions:
        print(prediction.verb, prediction.dependency_type, score)

    print("\nTesting Greedy Chain Generation")
    chain = [Event("know", "I", "nsubj")]
    print("seed event: ", chain[0])
    for i in range(3):
        prediction = predict_events(chain, document, n=1, embedding=EMBEDDING)[0]
        chain.append(prediction)
        print(prediction.verb, prediction.dependency_type)

def test_cloze():
    print("\nTesting Cloze Task")
    print("# of possible events: ", len(document.events))

    print("\nnyt narrative chains: ")
    accuracy = score_predictions(nyt_test_pairs, document, embedding=EMBEDDING)
    print("model position average: ", accuracy)

    print("\ncommon sense narrative chains")
    accuracy = score_predictions(common_sense_test_pairs, document, embedding=EMBEDDING)
    print("model position average: ", accuracy)

"""Test Runner"""
# test_prior() 
# test_joint()
# test_pmi()
test_prediction()
test_cloze()
