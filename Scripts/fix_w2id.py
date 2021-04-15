import json

with open("word_to_id.json", "r+") as word2id:
    w2id = json.load(word2id)

with open("fixed_word_to_id.json", "w") as fixed:

    res = dict((v,k) for k,v in w2id.items())
    json.dump(res, fixed)