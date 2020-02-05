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


def text_tokening(corporate_profile_text, tokenizer=tokenizer, verbose=0):
    # Sentence tokenization
    sent = sent_tokenize(corporate_profile_text)

    # Remove unwanted characters (this function should be factored out later)
    for i, s in enumerate(sent):
        sent[i] = s.replace('\n', '')

    sent_token_id = []
    for s in sent:
        sent_token_id.append((tokenizer.encode(s, pad_to_max_length=tokenizer.max_len)))

    if verbose == 1: # for checking
        # Turn index back to tokens
        tokens = []
        for i in range(len(sent_token)):
            tokens.append(tokenizer.convert_ids_to_tokens(sent_token[i]))

        # Print out token and id pair for checking
        for z in list(zip(sent_token, tokens)):         # loop over each sentence
            for i in range(len(z[0])):                  # loop over each tokens and ids
                print(z[0][i], z[1][i])

    return sent_token_id


def company2vector(sent_token_id, model):
    # Turn corperate Profile to vector

    input2model = torch.tensor(sent_token_id)
    outputs = model(input2model)

    # Take the last hidden layer outputs and average over sentence and words
    company_vector = torch.mean(outputs[0], dim=0)
    company_vector = torch.mean(company_vector, dim=0)

    # to numpy
    company_vector = company_vector.detach().numpy()

    return company_vector
