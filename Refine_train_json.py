import os
import json
import numpy as np
import nltk
sentences = {}
with open("new_train.json", "r+") as w:
        x = json.load(w)
        keys = list(x.keys())

        #for each in x.items():
        tokens = [nltk.tokenize.word_tokenize(str([keys[index]]["sentences"]).lower()) for index in np.arange(len(x.keys()))]
        print(tokens)
        #print(len(sentences["v_QOlSCBRmfWY"]["sentences"]))


