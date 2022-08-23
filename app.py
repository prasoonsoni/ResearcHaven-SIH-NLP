# !pip install nltk
# !pip install pandas
# !pip install numpy
# !pip install spacy
# !pip install ahpy

import statistics
from difflib import SequenceMatcher
from googleapiclient.discovery import build
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
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import string
from array import array
import nltk
from keybert import KeyBERT
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

my_api_key = "AIzaSyDaS_yp6jSkBET9Z5ozTpjfFtb6C2pvYB8"
my_cse_id = "037c2615c9b2e4e0f"


def research_plag_check(sus, og):
    sus_reference = sus['sus_reference']
    og_reference = og['sus_reference']
    similarity_ratio = SequenceMatcher(
        None, sus_reference, og_reference).ratio()
    return {"references_plag_check": similarity_ratio}


def google_search_result(sus):
    sus_title = sus['sus_title']
    sus_abstract = sus['sus_abstract']
    sus_keywords = sus['sus_keywords']
    sus_introduction = sus['sus_introduction']
    sus_proposed_method = sus['sus_proposed_method']
    sus_evaluation_result = sus['sus_evaluation_result']
    sus_conclusion = sus['sus_conclusion']

    sus = sus_title + ' ' + sus_abstract + ' ' + sus_keywords + ' ' + sus_introduction + \
        ' ' + sus_proposed_method + ' ' + sus_evaluation_result + ' ' + sus_conclusion

    def google_search(searching_for, api_key, cse_id, **kwargs):
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=searching_for, cx=cse_id, **kwargs).execute()
        return res

    def snippet_confidence(web_snippet, orig_chunk):
        web_snippet = web_snippet.replace('\n', '')
        orig_chunk = orig_chunk.replace('\n', '')
        match = SequenceMatcher(None, web_snippet, orig_chunk).find_longest_match(
            0, len(web_snippet), 0, len(orig_chunk))
        match = web_snippet[match.a: match.a + match.size]
        diff = round(len(match) / len(web_snippet), 2)
        return diff

    def calculate_score(confidence):
        mean = round(statistics.mean(confidence), 2)
        print('Average Score: ', mean)
        if 1.0 in confidence:
            return 1
        elif mean >= 0.50:
            return mean
        else:
            return mean

    data = sus
    chunks = list()
    end = 33
    start = 0
    while end < len(data):
        chunk = ' '.join(data[start:end])
        chunks.append(chunk)
        end = end + 33
        start = start + 33
        if end > len(data):
            end = len(data)
            chunk = data[start:end]
            chunks.append(chunk)
    confidence = []
    itr = 1
    for chunk in chunks:
        response = google_search(str(chunk), my_api_key, my_cse_id)
        num_results = response.get('searchInformation').get('totalResults')
        if num_results != '0':
            for item in response.get('items'):
                web_snippet = ''.join(item['snippet'][0:203])
                confidence.append(snippet_confidence(web_snippet, str(chunk)))
        itr = itr + 1
    return calculate_score(confidence)


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

    return frequent_words, frequent_bigrams, frequent_trigrams


def taking_input(user_file, db_file):
    sus_term, sus_bigram, sus_trigram = preprocess(user_file)
    db_term, db_bigram, db_trigram = preprocess(db_file)

    # return intersection of sus and db
    return len(list(set(sus_term).intersection(db_term))), len(list(set(sus_bigram).intersection(db_bigram))), len(list(set(sus_trigram).intersection(db_trigram)))


def printing_similarity(og, sus, type):
    if type == 1:
        # join values of sus dictionary to str
        sus = ' '.join(sus.values())
        sus_words = sus.split()
        suslen = len(sus_words)

        sus_title = ' '.join(sus_words[0:20])
        fivepercent = int(suslen*5/100)
        sus_abstract = ' '.join(sus_words[20:fivepercent])

        kw_model = KeyBERT()

        keywords = kw_model.extract_keywords(sus, keyphrase_ngram_range=(
            1, 3), stop_words="english", highlight=False, top_n=10)
        keywords = ", ".join(list(dict(keywords).keys()))
        sus_keywords = keywords
        sevenpercent = int(suslen*7/100)
        sus_introduction = ' '.join(
            sus_words[fivepercent+1:fivepercent + sevenpercent])
        fortyfivepercent = int(suslen*45/100)
        sus_proposed_method = ' '.join(
            sus_words[fivepercent + sevenpercent + 1: fortyfivepercent + fivepercent + sevenpercent])
        thirtyeightpercent = int(suslen*38/100)
        sus_evaluation_result = ' '.join(
            sus_words[fortyfivepercent + fivepercent + sevenpercent + 1: thirtyeightpercent + fortyfivepercent + fivepercent + sevenpercent])
        sus_conclusion = ' '.join(
            sus_words[thirtyeightpercent + fortyfivepercent + fivepercent + sevenpercent + 1:])
    else:
        sus_title = sus['sus_title']
        sus_abstract = sus['sus_abstract']
        sus_keywords = sus['sus_keywords']
        sus_introduction = sus['sus_introduction']
        sus_proposed_method = sus['sus_proposed_method']
        sus_evaluation_result = sus['sus_evaluation_result']
        sus_conclusion = sus['sus_conclusion']

    og_title = og['og_title']
    og_abstract = og['og_abstract']
    og_keywords = og['og_keywords']
    og_introduction = og['og_introduction']
    og_proposed_method = og['og_proposed_method']
    og_evaluation_result = og['og_evaluation_result']
    og_conclusion = og['og_conclusion']

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
        print(temp1, temp2, temp3)
        C[param + '_' + noOfterms[0]] = temp1
        C[param + '_' + noOfterms[1]] = temp2
        C[param + '_' + noOfterms[2]] = temp3
    weights = {'title': 0.541, 'abstract': 0.118, 'keywords': 0.118, 'conclusion': 0.096,
               'proposed_method': 0.049, 'evaluation_result': 0.049, 'introduction': 0.03}
    weights2 = {'candidate_set': 0.047, '2termset': 0.19, '3termset': 0.763}
    weights2 = dict(reversed(list(weights2.items())))
    print(weights2)
    similarity_score = 0
    sus = sus_title + sus_abstract + sus_keywords + sus_introduction + \
        sus_proposed_method + sus_evaluation_result + sus_conclusion
    sus_words = sus.split()
    suslen = len(sus_words)
    for i in weights.keys():
        for j in weights2.keys():
            print("w: ", weights[i])
            print("w2: ", weights2[j])
            print("C: ", C[i+'_'+j])
            similarity_score += weights[i] * weights2[j] * C[i + '_' + j]
    wk = sum(weights.values()) + sum(weights2.values())
    print(wk)
    print(similarity_score)
    print(suslen)
    similarity_score = similarity_score / (0.05*suslen)
    print(similarity_score)
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
    type: int


@app.post("/test/")
def func():
    return {"message": "Hello World"}


@app.post("/checkplagiarism/")
async def mainfunction(info: Info):

    info = info.dict()

    print("-------------------------", info)
    # google_similarity_score = google_search_result(info['sus'])
    similarity_score = printing_similarity(info['og'], info['sus'], info['type'])
    # , "google_similarity_score": google_similarity_score
    # print(google_similarity_score)
    return {"similarity_score": similarity_score}

@app.post("/referencesplagiarism/")
async def func(info:Info):
    info = info.dict()
    score = research_plag_check(info['sus'], info['og']);
    return {"references_similarity_score": score}