# Unsupervised Learning of Narrative Event Chains
> Nathanael Chambers and Dan Jurafsky (2008)

An updated implementation of *Unsupervised Learning of Narrative Event Chains* by Chambers and Jurafsky (2008) as part of an independent study project at the University of Pennsylvania, Fall 2019. The overall goal of the project is to learn discrete representations of narrative knowledge through **Narrative Events** and orderings known as **Narrative Chains**. From the paper: "Since we  are  focusing  on  a  single  actor  in  this  study,  a narrative event is thus a tuple of the event and the typed dependency of the protagonist". 

## Quickstart
1. Install dependencies with `pip install -r requirements.txt`
2. Download spaCy model with `python install -m spacy en`
3. Train model and evaluate by calling `python src/index.py --train`
4. (Optional) Evaluate pre-existing model by calling `python src/index.py`

Disclaimer: the `neuralcoref` package has issues in certain architectures. These issues can be resolved by uninstalling the package and re-installing from distribution source using `pip install neuralcoref --no-binary neuralcoref`.

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

Extensions include:
1. Magnitude Embedding Library (`pymagnitude`)
2. Word2Vec Google-News Word Embedding Model 

## Data
The algorithm is being tested on a subset of the New York Times section of the Gigaword Corpus using the `agiga` maven package. Raw text from Gigaword can be extracted using the `agiga` package through Maven:

```
mvn exec:java -Dexec.mainClass="edu.jhu.agiga.AgigaPrinter" -Dexec.args="words ../gigaword/nyt_eng_200901.xml.gz"
```

## Testing and Evaluation
Evaluation is performed using the *Narrative Cloze* Evaluation Task for narrative coherence. Implementation can be found in `code/evaluation.py`. A narrative chain is provided to the task and an event is removed in order for the model to perform a prediction to be evaluated on. The aim of the task is to perform a fill-in-the-blanks task, which upon successful completion indicates the presence of coherent narrative knowledge by the model. Given of tuple list of `(chain, event)` where `chain` is missing the true prediction `event`, the evaluation module returns the average model position. The model position is defined as the true event's position in the model's ranked candidate outputs (lower is better).

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

This result of roughly a 1:2 ratio between average model position and total verb count is comparable to the results achieved in the original paper by Chambers and Jurafsky.

## Implementation Notes 
- verb space too large -> lemmatizing verbs before parsing
- events are similar to themselves -> removing seen verbs in chain from prediction candidates
- coreference resolution fails occasionally -> increase chunk size
- parsing is slow -> single grammatical pass and resolve entities ad-hoc
- coreference count computation is slow -> refactor to matrix implementation

## License
MIT License

Copyright (c) 2020 Arun Kirubarajan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
