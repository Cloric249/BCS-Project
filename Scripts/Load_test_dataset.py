import os
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
import json
import nltk
import string
from tqdm import tqdm

with open("COCO_Captions.json") as captions:
    caps = json.load(captions)



with open("filtered words.json") as filtered:
    fil = json.load(filtered)
coco_cap_len = open("COCO_train_lengths.json", "w")
cap_lengths = {}
dataDir='.'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
capFile='{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)
coco_cap = COCO(capFile)
imgIds = coco.getImgIds(imgIds = [])
annIds = coco_cap.getAnnIds(imgIds=[])

#coco.download("D:/Downloads/COCO 2017 Images 2", annIds)

#img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

#print(len(imgIds))
#x = 0
#for each in tqdm(imgIds):
#    try:
#        caption = coco_cap.loadAnns(each)
#        img_id = caption[0]['image_id']
#        cap_lengths[each] = [img_id, str(caption[0]['caption'])]
#    except:
#        x = x + 1
#print(x)
#json.dump(cap_lengths, caps)
#caps.close()
d = {}
for each in caps.keys():
    caption = caps[each][1]
    tkn_caps = nltk.word_tokenize(caption)
    cap = []
    for x in tkn_caps:
          cap.append(x.lower())
    d[each] = len(cap)

json.dump(d, coco_cap_len)
#I = io.imread(img['coco_url'])
#plt.axis('off')
#plt.imshow(I)
#plt.show()


#json.dump(d, f)
#f.close()