import json
import nltk

with open("COCO_Caption_Lenghts.json", "r+") as captions:
    caps = json.load(captions)
with open("word_to_id.json", "r+") as word2id:
    w2id = json.load(word2id)

    with open("caption_train_lengths.json", "w") as lengths:

        dict = {}
        for each in caps.keys():
            cap_len = caps[each]
            f_list = []
            list = []
            tkns = nltk.tokenize.word_tokenize(x)
            for each in tkns:
                if each.lower() in w2id.keys():
                    list.append(each)
                f_list.append(len(list))
            dict[id] = f_list

        json.dump(dict, lengths)


