"""
Created using elements from a tutorial to RNN translation in PyTorch by Sean Robertson
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

import re
import pickle

MAX_LENGTH = 20

data_fp = "data/eng-ger.txt"
lang_fp = "eng-ger.pkl"

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    return s.strip()

def readLangs(lang1, lang2, data_fp, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(data_fp , encoding='utf-8').\
        read().strip().split('\n')  

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH 


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(lang1, lang2, data_fp, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, data_fp)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = prepareData('eng', 'ger', data_fp)

with open(lang_fp, 'wb') as out:
    print("Pickling languages")
    pickle.dump(input_lang, out, pickle.HIGHEST_PROTOCOL)
    pickle.dump(output_lang, out, pickle.HIGHEST_PROTOCOL)