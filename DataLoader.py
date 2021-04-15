import ast
import json
import os
from random import randrange
import sys
import nltk
import torch
import torch.utils.data as data
from pycocotools.coco import COCO
import skimage.io as io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import string
import time
import cv2
sys.path.append('/opt/cocoapi/PythonAPI')


from Classifier import getFrames


with open("word_to_id.json", "r+") as word2id:
    w2id = json.load(word2id)

with open("COCO_Captions.json") as caps:
    COCO_captions = json.load(caps)

with open("COCO_train_lengths.json") as caps_len:
    caption_lengths = json.load(caps_len)





# Explanation of parameters:
# transform: Defines the transformations for frames for the network
# network_mode: Defines the mode the data loader is in. If 'train' then the network works off the training
# dataset: If 'test' then the network works off the supplied images/frames
# batch_size: Defines how many batches the data is split into.
# vocab_file: Defines the file that contains the unique words that make up the training caption dataset
# start_tkn, end_tkn, uknown_tkn: Define their respective token
def get_data_loader(transform, network_mode="train", batch_size=1, vocab_file="Filtered Words.txt",
                    start_tkn="<SOS>", end_tkn="<EOS>", num_of_workers=0):
    assert network_mode == "train" or "validate" or "test"
    assert os.path.exists(vocab_file)

    dataset = ActivityNetLoader(transform=transform,
                                network_mode=network_mode,
                                batch_size=batch_size,
                                vocab_file=vocab_file,
                                start_tkn=start_tkn,
                                end_tkn=end_tkn)

    batch = dataset.get_random_batch()
    init_sampler = data.sampler.SubsetRandomSampler(indices=batch)
    data_loader = data.DataLoader(dataset=dataset,
                                  num_workers=num_of_workers,
                                  batch_sampler=data.sampler.BatchSampler(sampler=init_sampler,
                                                                          batch_size=dataset.batch_size,
                                                                          drop_last=False))
    # TODO Look to create vocab file if something happens to packaged file


    return data_loader


# This method is purely for deciding the length of captions that will be used for
# the batch
def get_batch_lengths(mode):
    if mode == "train":
        dataset = caption_lengths
    if mode == "validate":
        dataset = val_cap_len
    vals = set()
    for each in dataset.values():
        vals.add(each)
    valid = False
    while valid == False:
        length = randrange(min(vals), max(vals))
        if length in vals:
            valid = True

    return length


class ActivityNetLoader(data.Dataset):
    def __init__(self, transform, network_mode, batch_size, start_tkn, end_tkn,
                 vocab_file):

        self.transform = transform
        self.network_mode = network_mode
        self.batch_size = batch_size
        self.vocab = (vocab_file, start_tkn, end_tkn)
        # load the image ids and the caption ids into memory
        dataDir = '.'
        dataType = 'train2017'
        coco_train_anns = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
        coco_train_caps = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
        self.coco_ann = COCO(coco_train_anns)
        self.coco_caps = COCO(coco_train_caps)
        self.img_ids =  self.coco_ann.getImgIds(imgIds=[])
        self.caps = self.coco_caps.getAnnIds(imgIds=[])



    # if network is in training then obtain the key frames that correspond the captions
    # this method goes through each of the key frame training folders, returns 3 frames from various parts of the video
    # and associates the frames with their corresponding caption
    def __getitem__(self, index):
        if self.network_mode == "train":
            index = int(index)
            # Get the frame
            img_id = COCO_captions[str(index)][0]
            id_len = len(str(img_id))
            to_add = 12 - id_len
            img_id = str(0)*to_add + str(img_id) + ".jpg"
            image_path = "D:/Downloads/COCO 2017 Images/" + img_id
            image = Image.open(image_path)
            image = image.convert('RGB')
            image = self.transform(image)
            # Get the caption
            caption = COCO_captions[str(index)][1]
            # Process the caption for training
            caption = nltk.word_tokenize(caption)
            lower_caption = []
            for word in caption:
                lower_caption.append(word.lower())
            tkn_caption = []
            tkn_caption.append(self.vocab[1])
            for word in lower_caption:
                    if word in w2id.keys():
                        tkn_caption.append(word)
            tkn_caption.append(self.vocab[2])
            caption_to_id = []
            for word in tkn_caption:
                if word in w2id.keys():
                    caption_to_id.append(w2id[word])
            caption = torch.Tensor(caption_to_id).long()


            return image, caption


        if self.network_mode == "test":
            batch = self.batch
            image = batch[0]
            captions = batch[1]
            image = Image.fromarray(image)
            image.show()
            image = image.convert('RGB')
            image = self.transform(image)
            print(image.shape)

            return image, captions





    def get_random_batch(self):
        if self.network_mode == "train":
            valid = False
            while valid != True:
                cap_length = get_batch_lengths(self.network_mode)
                captions = COCO_captions
                batch = []

                canditates = []


                for each in captions.keys():
                    prospect = []
                    for word in nltk.word_tokenize(captions[each][1]):
                        if word.lower() in w2id.keys():
                            prospect.append(word)
                    if len(prospect) == cap_length:
                        canditates.append(each)

                if len(canditates) > 0:
                    valid = True
                    for step in range(1, self.batch_size+1):
                        batch.append(canditates[randrange(0, len(canditates))])


            return batch




        if self.network_mode == "test":
            batch = []
            captions = []
            dataDir = '.'
            dataType = 'val2017'
            # file that contains the images
            annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
            # file that contains the captions associated with the images
            capFile = '{}/annotations/captions_{}.json'.format(dataDir, dataType)
            coco = COCO(annFile)
            coco_caps = COCO(capFile)
            # get a random image from the dataset
            imgIds = coco.getImgIds(imgIds=[])
            img_id = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            try:
                img = io.imread(img_id['coco_url'])
            except:
                time.sleep(1)
                img = io.imread(img_id['coco_url'])

            # get the captions for that image
            annIds = coco_caps.getAnnIds(imgIds=img_id['id']);
            anns = coco_caps.loadAnns(annIds)
            for each in anns:
                captions.append(each['caption'])
            # append the image and captions to the batch
            batch.append(img)
            batch.append(captions)
            self.batch = batch

            return batch





