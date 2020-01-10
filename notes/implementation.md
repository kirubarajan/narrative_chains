# Implementation
A document highlighting the implementation of the project.

## Format
Chains are in the form of *verb/dependency*. An example of a narrative chain pertaining to Microsoft looks like:

```
1 worked work pp
2 had have pp
3 hired hire subj
4 bringing bring subj
5 has have subj
6 hired hire subj
7 acquired acquire subj
8 taken take subj
9 coming come pp
10 has have subj

--------------------------
Microsoft actions
```

## Event Slot Similarity with Arguments
Scoring new event slot `f, g` against a chain of size `n` by summing over scores between all pairs.

```python
def chain_similarity(chain, f, g):
	sum([pmi((e, d), (f, g)) for (e, d) in chain])
```

where `chain` is a narrative chain, `f` is a verb with grammatical argument `g` and `pmi` returns the point-wise mutual information for (`e, d`) against (`f, g`).

## Evaluation
Evaluation is performed using Narrative Cloze, a fill in the blank method: choosing the correct ending the beginning of a narrative.

> "A commonly used evaluation is the ‘Narrative Cloze Test’ (Chambers and Jurafsky,  2008) in which a system predicts a held-out event (a verb and its arguments) given a set of observed events. For example, the following is one such test with a missing event:{X threw, pulled X, told X, ???, X completed." - Mostafazadeh et al.

> "As is often the case, several works now optimize to this specific test, achieving higher scores with shallow techniques. This is problematic because the models often are not learning common-sense knowledge, but rather how to beat the shallow test." - Mostafazadeh et al.