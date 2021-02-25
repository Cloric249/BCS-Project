import json

class Vocab():
    def __init__(self, start_tkn="<SOS>", end_tkn="<EOS>",
                 vocab_file="Filtered Words.txt"):

        self.start_tkn = start_tkn
        self.end_tkn = end_tkn
        self.vocab_file = vocab_file
    # method for creating the vocab file for the first time of recreating the file in case of
    # accidental deletion
    def create_vocab(self):
        # create a dictionary mapping each word from the vocab file to
        # a unique integer
        orignal_dict = {}
        tokenized_dict = {}
        id = 0

        #words = open("Filtered Words.txt", "r")
        with open(self.vocab_file) as VF:
            words = json.load(VF)
        #
        tokenized_dict[self.start_tkn] = id
        id = id + 1
        tokenized_dict[self.end_tkn] = id
        id = id + 1
        keys = words.keys()
        for key in keys:
            tokenized_dict[key] = id
            id = id + 1
        # dump the dictionary to a json file for later use
        with open("word_to_id.json", "w") as file:
            json.dump(tokenized_dict ,file)

    # method returns the previously created vocab file
    def get_vocab(self):
        with open("word_to_id.json", "r") as file:
            vocab = json.load(file)

        return vocab

if __name__ == '__main__':
    x = Vocab()
    x.create_vocab()