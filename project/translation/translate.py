"""
Created using elements from a tutorial to RNN translation in PyTorch by Sean Robertson
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""

import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import time
from nltk import tokenize #, download

SOS_token = 0
EOS_token = 1
device = "cpu"
MAX_LENGTH = 20

encoder_fp = "E:/PDP/PDP-mds/project/translation/en-de_trained_model/EN-DE-Encoder.pt"
decoder_fp = "E:/PDP/PDP-mds/project/translation/en-de_trained_model/EN-DE-Decoder.pt"
lang_fp = "E:/PDP/PDP-mds/project/translation/en-de_trained_model/eng-ger.pkl"

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.gru(embedded)
        return output, hidden
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden
    
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

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions
    
    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

encoder = torch.load(encoder_fp)
decoder = torch.load(decoder_fp)

with open(lang_fp, 'rb') as inp:
    input_lang = pickle.load(inp)
    output_lang = pickle.load(inp)

def indexesFromSentence(lang, sentence):
    temp = []
    for word in sentence.split(' '):
        if word in lang.word2index.keys():
            temp.append(lang.word2index[word])
    return temp

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def translate_custom(encoder, decoder, input_lang, output_lang, fp_in, fp_out):
    i = 0
    with torch.no_grad():
        with open(fp_in, 'r') as f:
            sentences = f.read()
        output = []
        sentences = tokenize.sent_tokenize(sentences)

        for sentence in sentences:
            input_tensor = tensorFromSentence(input_lang, sentence)

            encoder_outputs, encoder_hidden = encoder(input_tensor)
            decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden)

            _, topi = decoder_outputs.topk(1)
            decoded_ids = topi.squeeze()

            decoded_words = []
            for idx in decoded_ids:
                if idx.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                decoded_words.append(output_lang.index2word[idx.item()])
            decoded_words.pop(-1)
            punct = decoded_words[-1]
            with open(fp_out, "a", encoding="utf-8") as f:
                decoded_words[0] = decoded_words[0].capitalize()
                f.write(" ".join(decoded_words[:-1]) + punct + " ")
                i += 1
                if i % 3 == 0:
                    f.write("\n")
    '''
    #Optional prints to display the results in command line
    print('input =', sentence)
    print('output =', ' '.join(decoded_words))
    '''

#MBART implementation
def translate(fp_in, fp_out, lang):
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    lang_dict = {"pl": "pl_PL", "de": "de_DE", "ru": "ru_RU"}
    
    if lang not in lang_dict:
        return "Incorrect language selected"

    with open(fp_in, 'r') as f:
        sentences = f.read()

    output = []
    sentences = tokenize.sent_tokenize(sentences)
    for sentence in sentences:
        tokenizer.src_lang = "en_XX"
        encoded_hi = tokenizer(sentence, return_tensors="pt")
        generated_tokens = model.generate(
            **encoded_hi,
            forced_bos_token_id=tokenizer.lang_code_to_id[lang_dict[lang]]
        )
        output.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])

    with open(fp_out, "w", encoding="utf-8") as f:
        i = 0
        for sentence in output:
            f.write(sentence + " ")
            i += 1
            if i % 3 == 0:
                f.write("\n")
    

#translate(encoder, decoder, input_lang, output_lang, "test_file.txt", "test_file_out.txt")

