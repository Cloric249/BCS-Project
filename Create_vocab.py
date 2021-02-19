# the nltk defines a set of common words that can act as a filter known as stop words
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import json
from nltk.tokenize import word_tokenize
import string
from collections import Counter
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np


# nltk.download('stopwords')
# nltk.download('punkt')
s = set(stopwords.words('english'))
stop_filtered_captions = []

ids_directory = "D:/Downloads/captions/train_ids.json"
with open(ids_directory) as train_ids:
    ids = json.load(train_ids)



# we then load in the json file containing the descriptions
description_directory = "D:/Downloads/captions/train.json"
with open(description_directory) as train_desc:
    captions = json.load(train_desc)

for each in tqdm(captions):
   for sentence in captions[each]["sentences"]:
       tkn = word_tokenize(sentence)
       stop_filtered_captions.append([word for word in tkn if word not in string.punctuation])





stop_filtered_captions = list(chain(*stop_filtered_captions))

stop_filtered_captions = [word.lower() for word in stop_filtered_captions]
stop_filtered_captions = [word for word in stop_filtered_captions if len(word) > 1]
stop_filtered_captions = [word for word in stop_filtered_captions if word.isalpha()]

counter = Counter(stop_filtered_captions)

keep_list = []
topn = Counter(counter).most_common(50)
topn2 = Counter(counter).most_common(5000)

distilled_dict = {key:val for key, val in Counter(counter).items() if val >= 10}
print(len(distilled_dict))

f = open("Filtered Words.txt", "w")
f.write(json.dumps(counter))
f.flush()
f.close()

file = open("Fwords.txt", "w")

with open('Filtered Words.txt', "r+") as d:
  for line in d.readlines():
    line = line.replace(',',',\n')
    file.write(line)

file.flush()

words, count = zip(*topn)

plt.rcParams.update({'font.size': 7})
plt.bar(words, count)
plt.ylabel('Count')
plt.xlabel('Words')
plt.xticks(rotation=90)
plt.show()

