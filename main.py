# !pip install nltk
# !pip install pandas
# !pip install numpy
# !pip install spacy
# !pip install ahpy

from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS
import ahpy
import spacy
import numpy as np
import pandas as pd
import string
import re
from array import array
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


# perform preprocessing on input data to get cleaned data
def preprocess(input_file):
    # perform segmentation
    sentences = sent_tokenize(input_file)
    # perform tokenization
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]
    # perform stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_sentences = [[word for word in sentence if word not in stop_words]
                          for sentence in tokenized_sentences]
    # perform punctuation removal
    punctuations = string.punctuation
    filtered_sentences = [[word for word in sentence if word not in punctuations]
                          for sentence in filtered_sentences]
    # perform lowercasing
    filtered_sentences = [[word.lower() for word in sentence]
                          for sentence in filtered_sentences]
    # perform lemmatization and stemming
    lemmatizer = WordNetLemmatizer()
    stemmed_sentences = [[lemmatizer.lemmatize(
        word) for word in sentence] for sentence in filtered_sentences]
    # perform pos tagging
    pos_tagged_sentences = [nltk.pos_tag(sentence)
                            for sentence in stemmed_sentences]
    # perform bag of words
    bag_of_words = [[word for word, pos in sentence]
                    for sentence in pos_tagged_sentences]
    # convert bag_of_words to a list of words
    bag_of_words = [word for sentence in bag_of_words for word in sentence]
    # perform bigram and trigram generation
    bigrams = nltk.bigrams(bag_of_words)
    trigrams = nltk.trigrams(bag_of_words)

    # perform term frequency calculation
    term_frequency = {}
    for word in bag_of_words:
        if word in term_frequency:
            term_frequency[word] += 1
        else:
            term_frequency[word] = 1

    # perform term frequency caluclation for bigrams
    bigram_frequency = {}
    for bigram in bigrams:
        if bigram in bigram_frequency:
            bigram_frequency[bigram] += 1
        else:
            bigram_frequency[bigram] = 1

    # perform term frequency caluclation for trigrams
    trigram_frequency = {}
    for trigram in trigrams:
        if trigram in trigram_frequency:
            trigram_frequency[trigram] += 1
        else:
            trigram_frequency[trigram] = 1

    # perform term frequency sorting
    sorted_term_frequency = sorted(
        term_frequency.items(), key=lambda x: x[1], reverse=True)
    sorted_bigram_frequency = sorted(
        bigram_frequency.items(), key=lambda x: x[1], reverse=True)
    sorted_trigram_frequency = sorted(
        trigram_frequency.items(), key=lambda x: x[1], reverse=True)

    # extract words with frequency greater than 10
    frequent_words = [word for word,
                      frequency in sorted_term_frequency if frequency > 0]
    frequent_bigrams = [bigram for bigram,
                        frequency in sorted_bigram_frequency if frequency > 0]
    frequent_trigrams = [trigram for trigram,
                         frequency in sorted_trigram_frequency if frequency > 0]

    # generate synonyms for frequent words using wordnet and add them to frequent_words list
    syn_words = []
    for word in frequent_words:
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                syn_words.append(l.name())
    frequent_words = frequent_words + syn_words
    frequent_words = list(set(frequent_words))

    # # generate synonyms for frequent bigrams using wordnet and add them to frequent_bigrams list
    # syn_bigrams = []
    # for bigram in frequent_bigrams:
    #     for syn in wordnet.synsets(bigram[0]):
    #         for l in syn.lemmas():
    #             syn_bigrams.append((l.name(), bigram[1]))
    # frequent_bigrams = frequent_bigrams + syn_bigrams
    # frequent_bigrams = list(set(frequent_bigrams))

    # # generate synonyms for frequent trigrams using wordnet and add them to frequent_trigrams list
    # syn_trigrams = []
    # for trigram in frequent_trigrams:
    #     for syn in wordnet.synsets(trigram[0]):
    #         for l in syn.lemmas():
    #             syn_trigrams.append((l.name(), trigram[1]))
    # frequent_trigrams = frequent_trigrams + syn_trigrams
    # frequent_trigrams = list(set(frequent_trigrams))

    return frequent_words, frequent_bigrams, frequent_trigrams


def taking_input(user_file, db_file):
    sus_term, sus_bigram, sus_trigram = preprocess(user_file)
    db_term, db_bigram, db_trigram = preprocess(db_file)

    # return intersection of sus and db
    return len(list(set(sus_term).intersection(db_term))), len(list(set(sus_bigram).intersection(db_bigram))), len(list(set(sus_trigram).intersection(db_trigram)))


def printing_similarity(og, sus):
    og_title = og['og_title']
    og_abstract = og['og_abstract']
    og_keywords = og['og_keywords']
    og_introduction = og['og_introduction']
    og_proposed_method = og['og_proposed_method']
    og_evaluation_result = og['og_evaluation_result']
    og_conclusion = og['og_conclusion']

    sus_title = sus['sus_title']
    sus_abstract = sus['sus_abstract']
    sus_keywords = sus['sus_keywords']
    sus_introduction = sus['sus_introduction']
    sus_proposed_method = sus['sus_proposed_method']
    sus_evaluation_result = sus['sus_evaluation_result']
    sus_conclusion = sus['sus_conclusion']

    og = [og_title, og_abstract, og_keywords, og_introduction,
          og_proposed_method, og_evaluation_result, og_conclusion]
    sus = [sus_title, sus_abstract, sus_keywords, sus_introduction,
           sus_proposed_method, sus_evaluation_result, sus_conclusion]

    C = {}
    paper_parameters = ["title", "abstract", "keywords", "introduction",
                        "proposed_method", "evaluation_result", "conclusion"]
    noOfterms = ['candidate_set', '2termset', '3termset']

    for i, param in enumerate(paper_parameters):
        temp1, temp2, temp3 = taking_input(sus[i], og[i])
        C[param + '_' + noOfterms[0]] = temp1
        C[param + '_' + noOfterms[1]] = temp2
        C[param + '_' + noOfterms[2]] = temp3

    # pairwise_comparisons = {('title', 'abstract'): 7, ('title', 'keywords'): 7, ('title', 'introduction'): 9, ('title', 'proposed_method'): 8,
    #                         ('title', 'evaluation_result'): 8, ('title', 'conclusion'): 7,
    #                         ('abstract', 'keywords'): 1, ('abstract', 'introduction'): 5, ('abstract', 'proposed_method'): 3,
    #                         ('abstract', 'evaluation_result'): 3, ('abstract', 'conclusion'): 1,
    #                         ('keywords', 'introduction'): 5, ('keywords', 'proposed_method'): 3, ('keywords', 'evaluation_result'): 3,
    #                         ('keywords', 'conclusion'): 1,
    #                         ('introduction', 'proposed_method'): 0.5, ('introduction', 'evaluation_result'): 0.5, ('introduction', 'conclusion'): 0.33,
    #                         ('proposed_method', 'evaluation_result'): 1, ('proposed_method', 'conclusion'): 0.5,
    #                         ('evaluation_result', 'conclusion'): 0.5}

    # pairwise_comparisons2 = {('candidate_set', '2termset'): 0.14, (
    #     'candidate_set', '3termset'): 0.11, ('2termset', '3termset'): 0.14}
    # compare = ahpy.Compare(
    #     name='compare', comparisons=pairwise_comparisons, precision=3, random_index='saaty')
    # compare2 = ahpy.Compare(
    #     name='compare2', comparisons=pairwise_comparisons2, precision=3, random_index='saaty')

    weights = {'title': 0.541, 'abstract': 0.118, 'keywords': 0.118, 'conclusion': 0.096, 'proposed_method': 0.049, 'evaluation_result': 0.049, 'introduction': 0.03}
    weights2 = {'candidate_set': 0.047, '2termset': 0.19, '3termset': 0.763}
    weights2 = dict(reversed(list(weights2.items())))
    similarity_score = 0
    sus = sus_title + sus_abstract + sus_keywords + sus_introduction + \
        sus_proposed_method + sus_evaluation_result + sus_conclusion
    sus_words = sus.split()
    suslen = len(sus_words)
    for i in weights.keys():
        for j in weights2.keys():
            similarity_score += weights[i] * weights2[j] * C[i + '_' + j]
    wk = sum(weights.values()) + sum(weights2.values())
    similarity_score = similarity_score / (0.05*suslen)
    return similarity_score


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Info(BaseModel):
    og: dict
    sus: dict

@app.post("/test/")
def func():
    return {"message": "Hello World"}

@app.post("/checkplagiarism/")
async def mainfunction(info: Info):

    info = info.dict()

    print("-------------------------", info)
    similarity_score = printing_similarity(info['og'], info['sus'])
    return {"similarity_score": similarity_score}