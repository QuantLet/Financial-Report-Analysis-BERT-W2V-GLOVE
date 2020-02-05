from typing import Any

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

import numpy as np
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
# logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# % matplotlib inline

# Load pre-trained model tokenizer (vocabulary)
# max_sent_length = 512 # BERT Base
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_sent_length = tokenizer.max_len
print("Max number of tokens in one sentence:", tokenizer.max_len)


def load_model():
    # Load pretrain model of BERT
    model = BertModel.from_pretrained('bert-base-uncased')
    return model


def text_tokening(corperate_profile_text, tokenizer=tokenizer, verbose=0):
    # Sentence tokenization
    sent = sent_tokenize(corperate_profile_text)

    # Remove unwanted characters (this function should be factored out later)
    for i, s in enumerate(sent):
        sent[i] = s.replace('\n', '')

    sent_token = []
    for s in sent:
        sent_token.append((tokenizer.encode(s, pad_to_max_length=tokenizer.max_len)))

    if verbose == 1:
        # Turn tokens into index
        indexed_token = []
        for st in sent_token:
            indexed_token.append(tokenizer.convert_tokens_to_ids(st))

        for pairs in (list(zip(indexed_token, sent_token))):
            print(pairs)

    return sent_token


def company2vector(corperate_profile, model):
    # Turn corperate Profile to vector

    input2model = torch.tensor(indexed_token)
    outputs = model(input2model)

    # Take the last hidden layer outputs and average over sentence and words
    company_vector = torch.mean(outputs[0], dim=0)
    company_vector = torch.mean(company_vector, dim=0)

    # to numpy
    company_vector = company_vector.detach().numpy()

    return company_vector
