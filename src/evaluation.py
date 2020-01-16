# testing narrative cloze
testing_pairs = [
    ([("receive", "clients", "nsubj"), ("download", "clients", "dobj")], ("make", "clients", "nsubj")),
    ([("fled", "gelman", "nsubj"), ("found", "gelman", "nsubj")], ("take", "gelman", "nsubj")),
    ([("am", "i", "nsubj"), ("did", "i", "nsubj"), ("think", "i", "nsubj")], ('believe', "i", "nsubj")),
    ([("bought", "team", "dobj"), ("included", "team", "dobj")], ("take", "team", "nsubj")),
    ([("heard", "parents", "nsubj"), ("talking", "parents", "nsubj")], ("choose", "parents", "nsubj")),
    ([("buy", 'stock', 'dobj'), ("lend", 'money', 'dobj')], ("struggle", 'edison', 'nsubj')),
    ([("advocated", 'league', 'nsubj'), ("fought", 'league', 'nsubj')], ("withdraw", 'league', 'nsubj')),
    ([("was", 'cranston', 'nsubj'), ("spent", 'cranston', 'nsubj'), ("fight", 'cranston', 'nsubj')], ("raise", 'cranston', 'nsubj')),
    ([("have", 'administration', 'nsubj'), ("convinced", 'administration', 'nsubj'), ("look", 'administration', 'nsubj')], ("push", 'administration', 'nsubj')),
    ([('hug', 'father', 'dobj'), ('tell', 'father', 'dobj')], ('love', 'father', 'dobj')),
    ([('be', 'i', 'nsubj'), ('get', 'i', 'nsubj'), ('have', 'i', 'nsubj')], ('call', 'i', 'nsubj'))
]

def get_position(predictions, correct):
    for i in range(len(predictions)):
        if predictions[i][0] == correct[0]:
            return i + 1
    return len(predictions)

print("\nEvaluating Narrative Cloze Positions: ")
positions = list()
for chain, correct in testing_pairs:
    predictions = predict(chain)
    position = get_position(predictions, correct)
    positions.append(position)
    print("position: ", position)

# computing averages
average = sum(positions) / len(positions)
print("\naverage position: ", average)

adjusted_average = sum([x for x in positions if x != len(verbs)]) / len([x for x in positions if x != len(verbs)])
print("adjusted average position: ", adjusted_average)