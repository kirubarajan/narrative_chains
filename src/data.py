# read file and clean input
with open(INPUT_FILE) as f:
    text = " ".join(f.readlines()[13:-7])

# write events to output file
with open(OUTPUT_FILE, "w") as file:
    for event in ordered:
        file.write("\n" + str(event))

# TODO: pickle subjects, objects, coreference, total
class Model: pass
model = Model()
model.subjects, model.objects, model.coreference = dict(subjects), dict(objects), dict(coreference)
model.total, model.total_coreference = total, total_coreference

print("\nDumping Model")
with open("model.pickle", "wb") as file:
    pickle.dump(model, file)
print("successfully saved to model.pickle")