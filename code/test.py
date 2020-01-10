"""Script for testing document parsing and narrative chain creation"""

from data import build_loader
from parse import parse_document
from models import Event, event_prob, joint_event_prob, pmi_approx, predict_events
from evaluation import predict_blank, score_predictions, get_position

"""Data Pre-Processing"""
# file_path = ""
# loader = build_loader(file_path)
# loader.sanity_check()

EXAMPLE = "Kevin joined the army. Kevin served the army. Kevin oversaw the army. Kevin resigned from the navy." 
MAX_LENGTH = 1000000
# text = loader.get_text()[:MAX_LENGTH]
text = EXAMPLE

document = parse_document(text)
events = list(document.events.items())

def test_prior():
    print("testing prior probability of each event")
    for event, count in events:
        print(event, event_prob(event, document))
    print(len(events), len(document.verbs), len(document.subjects), len(document.dependencies))  

def test_joint(): 
    print("\ntesting joint probability of events")
    event1, event2 = events[0][0], events[1][0]
    print(event1, "|", event2, "|", joint_event_prob(event1, event2, document))

    event1, event2 = events[1][0], events[2][0]
    print(event1, "|", event2, "|", joint_event_prob(event1, event2, document))

    event1, event2 = events[2][0], events[0][0]
    print(event1, "|", event2, "|", joint_event_prob(event1, event2, document))

def test_pmi():
    print("\ntesting pointwise mutual information approximation of events")
    event1, event2 = events[0][0], events[1][0]
    print(event1, "|", event2, "|", pmi_approx(event1, event2, document))

    event1, event2 = events[1][0], events[2][0]
    print(event1, "|", event2, "|", pmi_approx(event1, event2, document))

    event1, event2 = events[2][0], events[0][0]
    print(event1, "|", event2, "|", pmi_approx(event1, event2, document))

def test_prediction():
    print("\ntesting event prediction")
    event1, event2 = Event("Somebody", "joined", "navy", "dobj"), Event("Somebody", "served", "navy", "dobj")

    print(event_prob(event1, document))
    print(event_prob(event2, document))
    # print(joint_event_prob(event1, events[1][0], document))
    print(predict_events([event1, event2], document, embedding=True)[0])

def test_cloze():
    print("\ntesting cloze task")
    event1, event2 = Event("Somebody", "joined", "navy", "dobj"), Event("Somebody", "served", "navy", "dobj")
    event3 = Event("Somebody", "oversaw", "navy", "dobj")
    chain = [event1, event2, event3]
    
    predictions, correct = predict_blank(chain, document, 0)
    print("model position: ", get_position(predictions, correct))

    # for simplicity, assume all dependency types are direct objects
    chain1 = [Event("I", "wrote", "poem", "dobj"), Event("I", "presented", "poem", "dobj")]
    chain2 = [Event("Kevin", "started", "game", "dobj"), Event("Kevin", "played", "game", "dobj")]
    chain3 = [Event("He", "jumped", "up", "dobj"), Event("He", "fell", "down", "dobj")]

    test_pairs = [(chain1, Event("I", "won", "award", "dobj")), (chain2, Event("Kevin", "won", "game", "dobj")), (chain3, Event("He", "hurt", "himself", "dobj"))]
    accuracy = score_predictions(test_pairs, document)
    print("model position average: ", accuracy)

"""Test Runner"""
# insert desired tests here