import torch
import torch.nn as nn
from torchvision import transforms
from CNN import EncoderCNN, DecoderRNN
from Tokenization import ActivityNetLoader, get_data_loader
import torch.utils.data as data
import math
import json



class Training():
    def __init__(self):
        # prep the data to be passed through the model
        self.training_transform = transforms.Compose([transforms.Resize(256),
                                                 transforms.RandomCrop(224),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                                      (0.229, 0.224, 0.225))])

        self.data_batch_size = 32
        self.epochs = 3
        self.embed_size = 300
        self.hidden_size = 512
        self.log = "training_log.txt"
        with open("word_to_id.json") as vocab:
            x = json.load(vocab)
        print("Loading captions...")
        self.vocab_size = len(x.keys())
        self.train_data_loader = get_data_loader(transform=self.training_transform,
                                            network_mode="train",
                                            batch_size=self.data_batch_size)
        # obtain the number of captions
        self.num_of_captions = 0
        with open("new_train.json", "r+") as w:
            caps = json.load(w)
            for caption in caps.items():
                caption_length = len(caption[1]["sentences"])
                self.num_of_captions = self.num_of_captions + caption_length

    def setModel(self):
        self.encoderCNN = EncoderCNN(self.embed_size)
        self.decoderRnn = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoderCNN.to(self.device)
        self.decoderRnn.to(self.device)

        self.loss = nn.CrossEntropyLoss().cuda()
        self.params = list(self.encoderCNN.embed.parameters()) + list(self.decoderRnn.parameters())
        self.optimiser = torch.optim.Adam(params=self.params, lr=0.001)
        self.total_steps =  math.ceil(self.num_of_captions/self.data_batch_size)

    def train(self):
        for epoch in range(1, self.epochs+1):
            for step in range(1, self.total_steps+1):
                batch = self.train_data_loader.dataset.get_random_batch()
                sampler = data.sampler.SubsetRandomSampler(indices=batch)
                self.train_data_loader.batch_sampler.sampler = sampler

                caption_frames, tknized_captions = next(iter(self.train_data_loader))
                caption_frames = caption_frames.to(self.device)
                tknized_captions = tknized_captions.to(self.device)
                self.encoderCNN.zero_grad()
                self.decoderRnn.zero_grad()
                cap = 0
                for frame in caption_frames:
                    features = self.encoderCNN(frame)
                    output = self.decoderRnn(features, tknized_captions[cap])
                    loss = self.loss(output.view(-1, self.vocab_size), tknized_captions[cap].view(-1))
                    loss.backward()
                    self.optimiser.step()
                    cap = cap + 1

                print(loss.item())

if __name__ == "__main__":
    model = Training()
    model.setModel()
    model.train()









