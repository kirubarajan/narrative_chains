# Unsupervised Learning of Narrative Event Chains
> Nathanael Chambers and Dan Jurafsky (2008)

An updated implementation of *Unsupervised Learning of Narrative Event Chains* by Chambers and Jurafsky (2008) as part of an independent study project at the University of Pennsylvania, Fall 2019. The overall goal of the project is to learn discrete representations of narrative knowledge through **Narrative Events** and orderings known as **Narrative Chains**. From the paper: "Since we  are  focusing  on  a  single  actor  in  this  study,  a narrative event is thus a tuple of the event and the typed dependency of the protagonist". 

## Quickstart
1. Install dependencies with `pip install -r requirements.txt`
2. Download spaCy model with `python install -m spacy en`
3. Run project by calling `python src/index.py` to use pickled model (optional `--train` flag to train and save model)

## Examples
Examples of identified narrative events in the format `(subject, verb, dependency, dependency_type, probability)`:

```
you kiss girl dobj 0.00023724792408066428
that enables users dobj 0.00023724792408066428
God bestows benefaction dobj 0.00023724792408066428
Astronomers observed planets dobj 0.00023724792408066428
```

Examples of generated narrative chains (using a Greedy Decoding strategy):

(Embedding-Similary Based)
```
seed event: play I dsubj -> I play
score nsubj -> I score
win nsubj -> I win
beat nubj -> I beat
```

(Pointwise Mutual Information Approximation Based)
``` seed event:  go I nsubj -> I go
get nsubj -> I get
do nsubj -> I do
want nsubj -> I want
```

## Updates and Extensions
This implementation of (Chambers and Jurafsky, 2008) uses updated libaries, classes and functions. Written in Python, using the Stanford CoreNLP library (updated dependency parsing from transition model to neural-based Universal Dependencies) as well as the SpaCy pipeline (with extensions from HuggingFace). 

The following libraries are used throughout the study:
1. Stanford CoreNLP Python Implementation (`stanfordnlp`)
2. SpaCy Dependency Parser (`spacy`)
3. HuggingFace Neural Coreference Resolution (`neuralcoref`)

Future work and extensions include:
1. Magnitude Embedding Library (`pymagnitude`)
2. Word2Vec Google-News Word Embedding Model 

## Representation
Event chain definition is available in `code/models.py` as well as implementations of Point-Wise Mutual Information approximation of event chains. The `Event` type supports various native Python functionality such as `hash`, `str` and `eq`. Events can be instantiated in Python as following:

```python
from models import Event

event = Event("Arun", "finished", "project")
print(event == Event("Arun", "finish", "project")) # -> true
```

The parsing module implementation is available at `code/parsing.py`, which 1) parses linguistic features (e.g. parts-of-speech, noun phrase chunks) and 2) parses plaintext into a `Document` object `document` which contains a vocabulary of the subjects/verbs/dependencies and a frequency dictionary of `Events`. The `Document` class serves as the trained model. Point-wise Mutual Information can be computed as follows:

```python
from models import ppmi_approx

print(ppmi_approx(event1, event2, document))
```

Note: log probabilities are used in the Pointwise Mutual Information approximation to mitigate numerical underflow. Currently, co-referring entities are resolved using exact string matches.

## Data
The algorithm is being tested on a subset of the New York Times section of the Gigaword Corpus using the `agiga` maven package. The data is loaded using a custom `DataLoader` class, which supports cleaning output from the `agiga` package as well as other quality-of-life Python functions (e.g. `sanity_check()`, `len()`).

A Data Loader can be instantiated using the `build_loader()` helper as follows:

```python
from data import build_loader

file_path = "path_to_file"
loader = build_loader(file_path)
loader.sanity_check()
```

Raw text from Gigaword can be extracted using the `agiga` package through Maven:

```
mvn exec:java -Dexec.mainClass="edu.jhu.agiga.AgigaPrinter" -Dexec.args="words ../gigaword/nyt_eng_200901.xml.gz"
```

## Testing and Evaluation
Testing can be performed by running `python code/test.py`. For the example text of `"Kevin joined the army. Kevin served the army. Kevin oversaw the army."`, the output yields:

```
testing prior probability of each event
Kevin joined army 0.3333333333333333
Kevin served army 0.3333333333333333
Kevin oversaw army 0.3333333333333333

testing joint probability of events
Kevin joined army | Kevin served army | 0.1111111111111111
Kevin served army | Kevin oversaw army | 0.1111111111111111
Kevin oversaw army | Kevin joined army | 0.1111111111111111

testing pointwise mutual information approximation of events
Kevin joined army | Kevin served army | -2.3083356884473307
Kevin served army | Kevin oversaw army | -2.3083356884473307
Kevin oversaw army | Kevin joined army | -2.3083356884473307
```

The exact output depends on the tests configured under `"""Test Runner"""`. Note the symmetry of calculations since each discrete event shares the same coreferring (i.e. exact match) verb arguments.

Evaluation is performed using the *Narrative Cloze* Evaluation Task for narrative coherence. Implementation can be found in `code/evaluation.py`. A narrative chain is provided to the task and an event is removed in order for the model to perform a prediction to be evaluated on. The aim of the task is to perform a fill-in-the-blanks task, which upon successful completion indicates the presence of coherent narrative knowledge by the model. Given of tuple list of `(chain, event)` where `chain` is missing the true prediction `event`, the function `score_predictions` returns the average model position. The model position is defined as the true event's position in the model's ranked candidate outputs (lower is better).

An example of the Narrative Cloze output (for annotated New York Times):

```
# of verbs:  2286
total coreference count:  675950

Narrative Cloze Positions:
position:  2279
position:  2283
position:  713
position:  2167
position:  880
position:  109
position:  2048
position:  448
position:  634
position:  12
position:  2118

average position:  1244.6363636363637
```

## Changelog 
- lemmatizing verbs before parsing
- removing seen verbs in chain from prediction candidates

## License
MIT License

Copyright (c) 2020 Arun Kirubarajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
