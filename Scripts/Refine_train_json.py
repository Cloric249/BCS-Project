import json
import nltk
import string
sentences = {}
with open("COCO_Captions.json") as w:
    x = json.load(w)
d = {}
filtered = open("filtered words.json", "w")
for each in x.keys():
    sentence = nltk.word_tokenize(x[each])
    for each in sentence:
        if each not in string.punctuation:
            if each.lower() not in d.keys():
                d[each.lower()] = 1
            else:
                val = d[each.lower()]
                val = val + 1
                d[each.lower()] = val
#deleting = []
#for each in d.keys():
#    if d[each] <= 5:
#        deleting.append(each)

#for each in deleting:
#    del d[each]

json.dump(d, filtered)
filtered.close()

