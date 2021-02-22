import nltk
import torch
import os
import torch.utils.data as data
import ast
import json
from .Classifier import getFrames

with open("video_ids", "r+") as video_ids:
    ids = video_ids.read()

with open("new_train.json", "r+") as vCaptions:
    video_captions = json.load(vCaptions)

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
                    start_tkn="<start>",end_tkn="<end>"):


    assert network_mode == "train" or "test"
    assert os.path.exists(vocab_file)

    dataset = ActivityNetLoader(transform=transform,
                                network_mode=network_mode,
                                batch_size=batch_size,
                                vocab_file=vocab_file,
                                start_tkn=start_tkn,
                                end_tkn=end_tkn)

    #TODO Look to create vocab file if something happens to packaged file

class ActivityNetLoader(data.Dataset):
    def __init__(self, transform, network_mode, batch_size, start_tkn, end_tkn,
                 caption_file, vocab, img_folder):

        self.transform = transform
        self.network_mode = network_mode
        self.batch_size = batch_size
        self.vocab = (vocab, start_tkn, end_tkn, caption_file)
        self.img_folder = img_folder



    # if network is in training then obtain the key frames that correspond the captions
    # this method goes through each of the key frame training folders, returns 3 frames from various parts of the video
    # and associates the frames with their corresponding caption
    def __getitem__(self, index):
        if self.network_mode == "train":
            id = ids[index]
            captions = video_captions[id]["sentences"]
            caption_size = len(captions)
            caption_frames = getFrames(id, caption_size)
            i = 0
            # transform the image for preprocessing
            for frame in caption_frames:
                frame = self.transform(frame)
                caption_frames[i] = frame
                i = i + 1

            # tokenize the captions
            tknized_captions = []
            for caption in captions:
                cap = []
                tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                # append the start token
                cap.append(self.vocab[1])
                # appe
                cap.extend([token for token in tokens])
                cap.append(self.vocab[2])
                cap = torch.Tensor(cap).long()
                tknized_captions.append(cap)

            return caption_frames, tknized_captions

            #TODO caption for testing mode








