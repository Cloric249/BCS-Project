import os
import json
sentences = {}
with open("new_train.json", "r+") as w:
        x = json.load(w)

        for each in x.items():
            print(each[0])
            sentences[str(each[0])] = {"sentences": each[1]["sentences"]}

        print(len(sentences["v_QOlSCBRmfWY"]["sentences"]))


