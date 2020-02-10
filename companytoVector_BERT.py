from typing import Any

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

import numpy as np
import pandas as pd

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

    if verbose > 0:  # for checking
        # Turn index back to tokens
        tokens = []
        for i in range(len(sent_token_id)):
            tokens.append(tokenizer.convert_ids_to_tokens(sent_token_id[i]))
        return sent_token_id, list(zip(sent_token_id, tokens))
    if verbose > 1:  # Print out token and id pair for checking
        for z in list(zip(sent_token_id, tokens)):  # loop over each sentence
            for i in range(len(z[0])):  # loop over each tokens and ids
                print(z[0][i], z[1][i])

        return sent_token_id, list(zip(sent_token_id, tokens))

    return sent_token_id


def company2vector(sent_token_id, model, average_over_sent=True):
    # Turn corporate Profile to vector
    input2model = torch.tensor(sent_token_id)
    outputs = model(input2model)

    # Take the last hidden layer outputs and average over sentence and words
    if average_over_sent:
        company_vector = torch.mean(outputs[0], dim=0)
        company_vector = torch.mean(company_vector, dim=0)

    else:
        company_vector = outputs[0]

    # to numpy
    company_vector = company_vector.detach().numpy()

    return company_vector


def vector_of_each_whole_token(corporate_profile_tokens, result):
    """
  Inputs:
  cororate_profile_token: The second output of text_tokening()
  result: The last layer of BERT, output of company_vector()

  Output:
  Pandas DataFrame with sentence ids, token ids, and token vectors
  """

    # Indexing subwords
    R = []  # Sentence level
    for s in range(len(corporate_profile_tokens[1][:])):
        r = []  # Token level
        i = -1
        for t in corporate_profile_tokens[1][s][1]:  # [second output][sent][the tokens]
            if t.startswith("##"):
                r.append(i)
            else:
                i += 1
                r.append(i)
        R.append(r)

    # Taking mean of sub words
    corrected = []
    sent_text_list = []
    for s, r in enumerate(R):  # For each sentence in last layer
        sent_vector = []
        for n in np.unique(r):  # For each token in sentences
            mask = np.array(r) == n
            temp = np.mean(result[s:s + 1, mask, :], axis=1, keepdims=True)  # Taking mean over sub words
            sent_vector.append(temp)
        sent_vector = np.concatenate(sent_vector, axis=1)
        corrected.append(sent_vector)

        sent_text = tokenizer.convert_tokens_to_string(corporate_profile_tokens[1][s][1])
        sent_text = sent_text.split(' ')
        sent_text_list.append(sent_text)

    # Create Pandas DataFrame
    pd_temp = []
    for ns, s in enumerate(sent_text_list):
        for nt, t in enumerate(s):
            if (t != '[CLS]') & (t != '[PAD]') & (t != '[SEP]'):
                pd_temp.append({'sentence_number': ns,
                                'token_number': nt,
                                'token': t, 'vector': corrected[ns][0, nt, :]})

    pd_temp = pd.DataFrame(pd_temp)
    return pd_temp
