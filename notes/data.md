# Data
> Explanations of various datasets and corpi used. Datasets not committed to Git.

## Corpus
A subset of the Gigaword Corpus is used for training and evaluation. The `agiga` maven package is used to parse the compressed xml resources. A pre-processor for raw `agiga` export (word pipeline) is defined in `code/data.py` as part of the `DataLoader` class.

## Example Narrative Chains
Narrative Chains trained on New York Times from the early 2000s that include at least 5 events. These chains are made public by Chambers and were used for development as well as a baseline.

### Format
Event format: id string lemma grammatical-function
Event example: 3 obtained obtain subj
