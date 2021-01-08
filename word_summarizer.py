from transformers import BertTokenizer
from transformers import TFBertModel
import tensorflow as tf
import numpy as np
import random
import os
import nltk
import spacy

def load_model_tokenizer_BERT():
    """
    Loads BERT model from local directory, if downloading, replace with bert-base-cased
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    tokenizer = BertTokenizer.from_pretrained(f'{dir_path}\\bert_model')
    model = TFBertModel.from_pretrained(f'{dir_path}\\bert_model', output_hidden_states = True)    
    return tokenizer, model

def average_hidden_states(sentence_token, model):
    """
    Returns a tensor of shape(768, ) which should be a semantic representation of the sentence.
    Averages over the hidden_states of the model
    """
    output = model(sentence_token)
    tup_hidden_states = output['hidden_states']
    #Using 2nd last to remove bias towards mlm and nsp
    average = tf.squeeze(tf.math.reduce_mean(tup_hidden_states[-2], axis = 1))
    return average

def get_bounds(sentence_embeddings):
    """
    Returns the minimum and maximum value of each axis/dimension of the sentence embeddings (Column)
    """
    output = []
    for i in range(sentence_embeddings.shape[1]):
        min_max_tup = (np.min(sentence_embeddings[:,i]), np.max(sentence_embeddings[:,i]))
        output.append(min_max_tup)
    return output

def generate_centroids(sentence_embeddings, num_centroids):
    """
    Generates 3 random centroids to initialize the k-means algorithm
    """
    centroid_list = []
    for i in range(num_centroids):
        centroid_coord = []
        min_max_lst = get_bounds(sentence_embeddings)
        for i in range(len(min_max_lst)):
            coord = random.uniform(min_max_lst[i][0], min_max_lst[i][1])
            centroid_coord.append(coord)
        centroid_list.append(np.array(centroid_coord))
    return centroid_list
    
def compute_distance(point1, point2):
    """
    Returns the euclidian distance between 2 points
    """
    return np.linalg.norm(point1 - point2)

def assign_embedding_to_centroid(embeddings, centroid_list):
    """
    For each sentence embedding, returns the closest centroid to it
    """
    num_sentences = embeddings.shape[0]
    embedding_with_nearest_centroid = []
    for i in range(num_sentences):
        min_dist_centroid = 0
        min_dist = 9999999999999999
        for j in range(len(centroid_list)):
            distance = compute_distance(embeddings[i], centroid_list[j])
            if distance < min_dist:
                min_dist = distance
                min_dist_centroid = j
        tup = (embeddings[i], min_dist_centroid)
        embedding_with_nearest_centroid.append(tup)
    return embedding_with_nearest_centroid

def shift_centroids(embedding_with_nearest_centroid, centroid_list):
    """
    For each centroid, shift the centroids to their new mean based on their nearest sentences
    """
    centroid_dic = {}
    updated_centroid_list = []
    for i in range(len(centroid_list)):
        centroid_dic[i] = []
    for i in range(len(embedding_with_nearest_centroid)):
        nearest_centroid = embedding_with_nearest_centroid[i][1]
        centroid_dic[nearest_centroid].append(embedding_with_nearest_centroid[i][0])
    for i in range(len(centroid_list)):
        if not centroid_dic[i]:
            updated_centroid_list.append(centroid_list[i])
            continue
        new_coord = np.mean(centroid_dic[i], axis = 0)    
        updated_centroid_list.append(new_coord)
    return updated_centroid_list

def k_means_pass(embeddings, num_iterations, num_centroids):
    loss_value = 999999999999
    best_centroid_list = None
    #Initialize 10 different set of centroids
    for i in range(10):
        centroid_list = generate_centroids(embeddings, num_centroids)
        iteration_loss_value = 0
        #left to compute minimal loss
        for j in range(num_iterations):
            embeddings_with_nearest_centroid = assign_embedding_to_centroid(embeddings, centroid_list)
            centroid_list = shift_centroids(embeddings_with_nearest_centroid, centroid_list)
        #Compute iteration loss value
        for i in range(len(embeddings_with_nearest_centroid)):
            iteration_loss_value += compute_distance(embeddings_with_nearest_centroid[i][0], centroid_list[embeddings_with_nearest_centroid[i][1]])
        if iteration_loss_value < loss_value:
            loss_value = iteration_loss_value
            best_centroid_list = centroid_list
    return best_centroid_list    

def get_best_sentences(centroid_list, embeddings):
    """
    Returns the sentence embeddings of the best sentences
    """
    best_sentences = []
    for coord in centroid_list:
        shortest_distance = 99999999
        best_curr_sentence = None
        for embedding in embeddings:
            if compute_distance(coord, embedding) < shortest_distance:
                shortest_distance = compute_distance(coord, embedding)
                best_curr_sentence = embedding
        best_sentences.append(best_curr_sentence)
    return best_sentences

def split_into_sentences(text):
    output = nltk.tokenize.sent_tokenize(text)
    return output

def sentence_summarizer(sentence_list, tokenizer, model, summary_length):
    token_lst = []
    for i in range(len(sentence_list)):
        output = tokenizer.encode(sentence_list[i], return_tensors = 'tf')
        token_lst.append(output)
    sentence_embeddings = []
    sentence_dict = {}
    for i in range(len(token_lst)): 
        averaged_outputs = average_hidden_states(token_lst[i], model)
        sentence_embeddings.append(averaged_outputs)
        sentence_dict[i] = averaged_outputs
    sentence_embeddings = np.array(sentence_embeddings)
    best_centroid_list = k_means_pass(sentence_embeddings, 50, summary_length)
    best_sentence_embeddings = get_best_sentences(best_centroid_list, sentence_embeddings)  

    output_sentence_lst = []
    for arr in best_sentence_embeddings:
        for key,val in sentence_dict.items():
            equal = arr == np.array(val)
            if equal.all():
                output_sentence_lst.append(key)

    output_sentence_lst.sort()
    summary = ""
    for i in output_sentence_lst:
        summary += sentence_list[i]
        summary += " "
    return summary

def NER_summary(summary):
    nlp = spacy.load('en_core_web_sm')
    filter = {'ents':['GPE', 'ORG', 'PERSON']}
    NER = nlp(summary)
    summary_lst = summary.split(" ")
    for i in range(len(summary_lst)):
        if summary_lst[i] in NER.ents:
            summary_lst[i] = "<strong>" + summary_lst[i] + "</strong>"
    return " ".join(summary_lst)


