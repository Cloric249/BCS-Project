import nltk
import torch
import os
import torch.utils.data as data
import ast
import json
from random import randrange
from Classifier import getFrames


with open("video_ids", "r+") as video_ids:
    ids = video_ids.read()

with open("new_train.json", "r+") as vCaptions:
    video_captions = json.load(vCaptions)

with open("word_to_id.json", "r+") as word2id:
    w2id = json.load(word2id)
sentences = {}
for each in video_captions.items():
    sentences[str(each[0])] = {"sentences": each[1]["sentences"]}


ids = ast.literal_eval(ids)




# Explenation of parameters:
# transform: Defines the transformations for frames for the network
# network_mode: Defines the mode the data loader is in. If 'train' then the network works off the training
# dataset: If 'test' then the network works off the supplied images/frames
# batch_size: Defines how many batches the data is split into.
# vocab_file: Defines the file that contains the unique words that make up the training caption dataset
# start_tkn, end_tkn, uknown_tkn: Define their respective token
def get_data_loader(transform, network_mode="train", batch_size=1, vocab_file="Filtered Words.txt",
                    start_tkn="<SOS>",end_tkn="<EOS>", num_of_workers=0):


    assert network_mode == "train" or "test"
    assert os.path.exists(vocab_file)

    dataset = ActivityNetLoader(transform=transform,
                                network_mode=network_mode,
                                batch_size=batch_size,
                                vocab_file=vocab_file,
                                start_tkn=start_tkn,
                                end_tkn=end_tkn)

    batch_ids = dataset.get_random_batch()
    init_sampler =  data.sampler.SubsetRandomSampler(indices=batch_ids)
    data_loader = data.DataLoader(dataset=dataset,
                                  num_workers=num_of_workers,
                                  batch_sampler=data.sampler.BatchSampler(sampler=init_sampler,
                                                                          batch_size=dataset.batch_size,
                                                                          drop_last=False))
    #TODO Look to create vocab file if something happens to packaged file

    return data_loader

class ActivityNetLoader(data.Dataset):
    def __init__(self, transform, network_mode, batch_size, start_tkn, end_tkn,
                vocab_file):

        self.transform = transform
        self.network_mode = network_mode
        self.batch_size = batch_size
        self.vocab = (vocab_file, start_tkn, end_tkn)
        self.list_of_ids = list(video_captions.keys())


    # if network is in training then obtain the key frames that correspond the captions
    # this method goes through each of the key frame training folders, returns 3 frames from various parts of the video
    # and associates the frames with their corresponding caption
    def __getitem__(self, index):
        if self.network_mode == "train":
            id = index
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
                cap2id = []
                tokens = nltk.tokenize.word_tokenize(str(caption).lower())

                # append the start token
                cap.append(self.vocab[1])
                # appe
                cap.extend([token for token in tokens])
                cap.append(self.vocab[2])
                for word in cap:
                    if word in w2id.keys():
                        val = w2id[word]
                        cap2id.append(val)
                print(cap2id)
                cap2id = torch.Tensor(cap2id).long()
                tknized_captions.append(cap2id)

            return caption_frames, tknized_captions

    def get_random_batch(self):
        batch_ids = []
        for x in range(1, self.batch_size):
            batch_ids.append(self.list_of_ids[randrange(len(self.list_of_ids))])

        return batch_ids

        #TODO caption for testing mode








