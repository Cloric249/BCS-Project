import nltk
import torch
import os
import torch.utils.data as data
import ast
import json

with open("video_ids", "r+") as video_ids:
    ids = video_ids.read()

with open("new_train.json", "r+") as vCaptions:
    video_captions = json.load()

sentences = {}
for each in video_captions.items():
    sentences[str(each[0])] = {"sentences": each[1]["sentences"]}


ids = ast.literal_eval(ids)




# Explenation of parameters:
# transform: Defines the transformations for frames for the network
# network_mode: Defines the mode the data loader is in. If 'train' then the network works off the training
# dataset. If 'test' then the network works off the supplied images/frames
# batch_size: Defines how many batches the data is split into.
# vocab_file: Defines the file that contains the unique words that make up the training caption dataset
# start_tkn, end_tkn, uknown_tkn: Define their respective token
def get_data_loader(transform, network_mode="train", batch_size=1, vocab_file="Filtered Words.txt",
                    start_tkn="<start>",end_tkn="<end>", unknown_tkn="<unknown>"):


    assert network_mode == "train" or "test"
    assert os.path.exists(vocab_file)

    #TODO Look to create vocab file if something happens to packaged file

class ActivityNetLoader(data.Dataset):
    def __init__(self, transform, network_mode, batch_size, start_tkn, end_tkn,
                 unknown_tkn, caption_file, vocab, img_folder):

        self.transform = transform
        self.network_mode = network_mode
        self.batch_size = batch_size
        self.vocab = (vocab, start_tkn, end_tkn, unknown_tkn, caption_file)
        self.img_folder = img_folder


    # if network is in training then obtain the key frames that correspond the captions
    # this method goes through each of the key frame training folders, returns 3 frames from various parts of the video
    # and associates the frames with their corresponding caption
    def getFrameCaption(self, index):
        if self.network_mode == "train":
            id = ids[index]
            captions = video_captions[id]["sentences"]
            if len(captions)





