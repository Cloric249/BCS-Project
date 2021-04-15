import json
import math
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms
import os
from Model import EncoderCNN, DecoderRNN
from DataLoader import get_data_loader
from pycocotools.coco import COCO
sys.path.append('/opt/cocoapi/PythonAPI')
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize.treebank import TreebankWordDetokenizer
import time
# nltk.download('wordnet')


class Training:
    def __init__(self):
        # prep the data to be passed through the model
        self.training_transform = transforms.Compose([transforms.Resize(256),
                                                      transforms.RandomCrop(224),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.485, 0.456, 0.406),
                                                                           (0.229, 0.224, 0.225))])

        self.data_batch_size = 1
        self.epochs = 5
        self.embed_size = 200
        self.hidden_size = 450
        self.log = open("COCO training log 5 epochs 0.001LR.txt", "w")
        with open("word_to_id.json") as vocab:
            x = json.load(vocab)
        print("Loading captions...")
        self.vocab_size = len(x.keys())
        self.mode = "test"
        self.train_data_loader = get_data_loader(transform=self.training_transform,
                                                 network_mode=self.mode,
                                                 batch_size=self.data_batch_size)
        # obtain the number of captions
        if self.mode == "train":
            with open("COCO_Caption_Lenghts.json", "r+") as w:
                caps = json.load(w)
                self.num_of_captions = len(caps.keys())

        if self.mode == "validate":
            with open("val_1.json", "r+") as w:
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
        if self.mode == "train":
            self.total_steps = math.ceil((self.num_of_captions / self.data_batch_size))

    def train(self):
        for epoch in range(1, self.epochs+1):
            for step in range(1, self.total_steps + 1):
                batch = self.train_data_loader.dataset.get_random_batch()
                sampler = data.sampler.SubsetRandomSampler(indices=batch)
                self.train_data_loader.batch_sampler.sampler = sampler

                frame, tknized_captions = next(iter(self.train_data_loader))
                self.encoderCNN.zero_grad()
                self.decoderRnn.zero_grad()
                print(frame.shape)
                frame = frame.to(self.device)

                caption = tknized_captions.to(self.device)
                features = self.encoderCNN(frame)
                output = self.decoderRnn(features, caption)
                loss = self.loss(output.view(-1, self.vocab_size), caption.view(-1))
                loss.backward()
                self.optimiser.step()
                stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (
                    epoch, self.epochs, step, self.total_steps, loss.item(), np.exp(loss.item()))
                print(stats)
                self.log.write(stats + "\n")
                self.log.flush()

                try:
                    torch.save(self.encoderCNN.state_dict(), "./Models/Encoder.pkl")
                    torch.save(self.decoderRnn.state_dict(), "./Models/Decoder.pkl")
                except:
                    time.sleep(1)
                    torch.save(self.encoderCNN.state_dict(), "./Models/Encoder.pkl")
                    torch.save(self.decoderRnn.state_dict(), "./Models/Decoder.pkl")

        self.log.close()

    def test(self):
        with open("word_to_id.json") as w2id:
            translate = json.load(w2id)

        test_log = open("COCO Test Log Model 2.txt", "w")
        encoder = EncoderCNN(self.embed_size)
        decoder = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size)
        # set the model to inference mode
        encoder.eval()
        decoder.eval()
        # load the weights
        encoder.load_state_dict(torch.load("./Models/Encoder.pkl"))
        decoder.load_state_dict(torch.load("./Models/Decoder.pkl"))
        # push to gpu
        encoder.to(self.device)
        decoder.to(self.device)
        blue_score = 0
        met_score = 0
        for step in range(1, 5000):
            batch = self.train_data_loader.dataset.get_random_batch()
            image, captions = next(iter(self.train_data_loader))
            image = image.to(self.device)
            features = encoder(image)
            output = decoder.interpret(features)
            sentence = []
            for each in output:
                if each in translate.values():
                        sentence.append(list(translate.keys())[list(translate.values()).index(each)])
                if each == 0:
                    sentence.remove("<SOS>")
                if each == 1:
                    sentence.remove("<EOS>")

            score = 0
            for each in captions:
                score = score + sentence_bleu(each, TreebankWordDetokenizer().detokenize(sentence))
            score = score/len(captions)
            blue_score = blue_score + score
            score = 0
            for caption in captions:
                temp_met_score = meteor_score(str(caption), TreebankWordDetokenizer().detokenize(sentence))
                score = score + temp_met_score
            score = score/len(captions)
            met_score = met_score + score
        blue_score = blue_score/5000
        met_score = met_score/5000
        test_log.write("BLUE SCORE: " + str(blue_score) + "\n")
        test_log.write("METEOR SCORE: " + str(met_score))

        test_log.flush()
        test_log.close()

    def convert_model(self):
        encoder = EncoderCNN(self.embed_size)
        decoder = DecoderRNN(self.embed_size, self.hidden_size, self.vocab_size)

        encoder.eval()
        decoder.eval()
        encoder.load_state_dict(torch.load("./Models/Encoder.pkl"))
        decoder.load_state_dict(torch.load("./Models/Decoder.pkl"))

        batch = self.train_data_loader.dataset.get_random_batch()
        sampler = data.sampler.SubsetRandomSampler(indices=batch)
        self.train_data_loader.batch_sampler.sampler = sampler

        frame, tknized_captions = next(iter(self.train_data_loader))

        input_cnn = torch.rand(1, 3, 224, 224)
        input_rnn = torch.rand(1, 1, 200)

        script_cnn = torch.jit.trace(encoder, input_cnn)
        script_rnn = torch.jit.trace(decoder, input_rnn, strict=False)

        script_cnn.save("Encoder.pt")
        script_rnn.save("Decoder.pt")




if __name__ == "__main__":
    model = Training()
    model.setModel()
    #model.train()
    model.test()
    #model.convert_model()
