import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

#device = torch.device('cuda:0')

# This is currently expermimentation with different models as a test of their performance
# This class defines the Encoder for the video captioning model
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resNet = models.resnet50(pretrained=True, progress=True)
        # number of input features for resNet model
        num_in_features = print("Input Features: ", resNet.fc.in_features)
        # number of output features for resNet model
        num_out_features = print("Output Features: ", resNet.fc.out_features)
        relu = nn.ReLU()
        # uses the GPU instead of the CPU for computing features
        #resNet = resNet.to(device)
        Loss = nn.CrossEntropyLoss()
        # remove fully connected layer
        for param in resNet.parameters():
            param.requires_grad_(False)

        modules = list(resNet.children())[:-1]
        self.resNet = (nn.Sequential(*modules))
        self.embed = nn.Linear(resNet.fc.in_features, embed_size)
        self.batch = nn.BatchNorm1d(embed_size, momentum= 0.1)
        self.embed.weight.data.normal_(0, 0.02)
        self.embed.bias.data.fill_(0)

    # forward propogation of features
    def forward(self, frames):
        features = self.resNet(frames)
        features = features.view(features.size(0), -1)
        features = self.batch(self.embed(features))

        features = features.unsqueeze(0)
        return features



# This class defines the RNN for the video captioning model
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_of_layers=1):
        super().__init__()
        # Defines the number of layers
        self.num_of_layers = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_of_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def interpret(self, input, states=None):
        output = []
        for each in range(1, 20):
            lstm_outputs, states = self.lstm(input, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            out = self.linear(lstm_outputs)
            last = out.max(1)[1]
            output.append(last.item())
            input = self.num_of_layers(last).unsqueeze(1)
        
        return output



