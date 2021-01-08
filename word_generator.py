from transformers import GPT2Tokenizer
from transformers import TFGPT2Model, TFGPT2LMHeadModel
import tensorflow as tf
import numpy as np
import os

#Install the GPT-2 model/tokenizer and load it into memory
def load_model_tokenizer_GPT2():
    """
    Loads GPT-2 model from local memory. Replace with gpt2
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tokenizer = GPT2Tokenizer.from_pretrained(f'{dir_path}\\gpt2_model')
    model = TFGPT2LMHeadModel.from_pretrained(f'{dir_path}\\gpt2_model')    
    return tokenizer, model

#Tokenizer any given text and return
def tokenize_text(tokenizer, text):
    #Using tensorflow backend
    #Removing space 
    if text[-1] == " ":
        text = text[:-1]
    tokenized = tokenizer.encode(text, return_tensors='tf')
    return tokenized

#Next Word algorithm
def next_word_prediction(tokenizer, model, text, num_results = 3):
    tokens = tokenize_text(tokenizer, text)
    output = model(tokens)
    #Returns the logits of predictions for the last word in the sequence
    next_word_logits = output.logits[:, -1, :]
    softmaxed_next_word = tf.nn.softmax(next_word_logits)
    most_likely_words = tf.math.top_k(softmaxed_next_word, num_results)
    prob_most_likely_words = np.array(most_likely_words.values).squeeze()
    index_most_likely_words = np.array(most_likely_words.indices).squeeze()
    prob_word_dic = {}
    for i in range(num_results):
        prob = prob_most_likely_words[i]
        word = tokenizer.decode(int(index_most_likely_words[i]))
        prob_word_dic["word" + str(i)] = word
    return prob_word_dic
