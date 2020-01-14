"""Model definitions for Event Chains and PMI calculations"""
from collections import defaultdict
import math
from pymagnitude import Magnitude

"""document class definition for managing corpus context"""
class Document:
    def __init__(self):
        self.left_events = list()
        self.right_events = list()
        self.verbs = set()
        self.dependencies = set()
        self.dependency_types = set()
        self.events = defaultdict(int)
        self.ordered_events = list()

"""event class definition for tracking event arguments and pos tags"""
class Event:
    def __init__(self, verb, dependency, dependency_type=None):
        self.verb = verb
        self.dependency = dependency
        self.dependency_type = dependency_type

    def __str__(self):
        return " ".join([self.verb, self.dependency, self.dependency_type])

    def __hash__(self):
        return hash(str(self.verb + " " + self.dependency))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

"""helper to compute count of two events sharing co-referring arguments"""
def count(event1, event2, document):
    # matches performed using dependency type
    frequency = 0
    """
    for event_1 in document.events:
        for event_2 in document.events:
            if event1.verb == event_1.verb and event2.verb == event_2.verb:
                if event1.dependency_type == event2.dependency_type:
                    frequency += 1
    return frequency
    """

    """
    for event in document.events:
        if event.verb == event1.verb or event.verb == event2.verb:
            # todo: replace with coreferring entity id
            if event.dependency_type == event1.dependency_type and event.dependency_type == event2.dependency_type:
                frequency += 1
    """

    for event in document.events:
        if event.dependency == event1.dependency and event.dependency == event2.dependency:
            frequency += document.events[event]
    return frequency

"""joint coreference probability (section 4)"""
def joint_event_prob(event1, event2, document):
    # marginalizing over each verb/dependency pair
    """
    total = 0
    for event in document.events:
        if event.dependency == event1.dependency and event.dependency == event2.dependency:
            total += document.events[event]
    
    if total == 0:
        return 0
    else:
    """
    return count(event1, event2, document) / document.total

"""coreference probability (section 4)"""
def event_prob(event, document):
    return document.events[event] / document.total

"""pointwise mutual information approximation"""
def pmi_approx(event1, event2, document):
    numerator = math.log(joint_event_prob(event1, event2, document) + 0.00000001)
    denominator = math.log(event_prob(event1, document) + 0.000001) + math.log(event_prob(event2, document) + 0.000001)
    result = math.exp(numerator - math.exp(denominator)) 
    return math.log(result) if result > 0 else 0 

"""predict next event given ordered list of events"""
def predict_events(chain, document, n=None, embedding=False, include_ranks=False):
    # iterate over every event in the document
    # compute sum of pmi between candidate and chain and take top n

    if n is None:
        n = len(document.events)
    if embedding: 
        vectors = Magnitude("data/GoogleNews-vectors-negative300.magnitude")

    scores = dict()

    for candidate in document.ordered_events:
        score = 0
        for event in chain:
            if embedding: similarity = vectors.similarity(candidate.verb, event.verb)
            else: similarity = pmi_approx(candidate, event, document)
            score += similarity
        scores[candidate] = score

    cleaned_scores = dict()
    verbs = set()
    for event in chain:
        verbs.add(event.verb)

    for candidate in scores:
        if candidate.verb not in verbs:
            cleaned_scores[candidate] = scores[candidate]
    
    ranked_scores = sorted(list(cleaned_scores.items()), key=lambda x: x[1], reverse=True)
    
    if include_ranks: return ranked_scores[:n]
    else: return [x[0] for x in ranked_scores[:n]]