import requests
import re
import os

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')

global index
index = 0


def extract(url):
    # print(url)
    global index                   # this anther way of scraping text,but extreme_Extraction() is the most accurte way of scraping
    response = requests.get(url=url).content
    soup = BeautifulSoup(response, 'lxml')
    title = soup.find('h1', {'class': "entry-title"})
    tag = soup.find('div', {'class': "td-post-content tagdiv-type"})
    if title is None and tag is None:
        title = soup.find('h1', {'class': "tdb-title-text"})
        tag = soup.find('div', {'class': "tdb-block-inner td-fix-index"})
        if title is None:
            print("Index : ", index, " ", url)
            return
        p = tag.find_all("p")
        l = [span.text.lower() for span in p]
        l.insert(0, title.text.lower())
        index += 1
        return ' '.join(l)
    p = tag.find_all("p")
    l = [span.text.lower() for span in p]
    l.insert(0, title.text.lower())
    index += 1
    return ' '.join(l)


def extreme_Extraction(url_list):
    driver = Chrome()
    text_list = []
    for url in tqdm(url_list, desc='extreme_Extraction', total=114):
        # url = url.encode('ascii', 'ignore').decode('unicode_escape')
        driver.get(url)
        tags = driver.find_elements(By.TAG_NAME, 'p')
        p_list = [i.text for i in tags]
        p_list.remove('© All Right Reserved, Blackcoffer(OPC) Pvt. Ltd')
        p_list.remove('Contact us: hello@blackcoffer.com')
        p_list.remove(
            'We provide intelligence, accelerate innovation and implement technology with extraordinary breadth and depth global insights into the big data,data-driven dashboards, applications development, and information management for organizations through combining unique, specialist services and high-lvel human expertise.')
        p_list = [sent for sent in p_list if sent != ""]
        if p_list[0] == "Sorry, but the page you are looking for doesn't exist.":
            p_list = None
            text_list.append(p_list)
            continue
        text_list.append(' '.join(p_list))
    driver.quit()
    return text_list


def filter(words):
    l = []
    for word in words:
        pattern2 = ' | '
        match = re.split(pattern2, word)
        if match:
            l.append(match[0])
            # print(match[0])
        else:
            print("No match found")
    return l


def transform_text(text):
    # text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        i.replace('?', '')
        i.replace('!', '')
        i.replace(',', '')
        i.replace('.', '')
        i.replace("'", "")
        i.replace("[", "")
        i.replace("]", "")

    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in STOPWORDS:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


p = set()


def syllable_count(words):
    pattern = "\w*[aeiou]\w*"
    if words is None:
        return []
    match = re.findall(pattern, words)  # getting list of words have vowels
    l = set(match)
    temp = set()
    for word in match:
        es_word = re.findall("\w+es$", word)
        if len(es_word) != 0:  # getting the words end with 'es' from the vowel list
            temp.update(es_word)
    l.difference_update(temp)  # updating the set by removing words end with 'es'
    temp.clear()
    for word in match:
        ed_word = re.findall("\w+ed$", word)
        if len(ed_word) != 0:  # getting the words end with 'ed' from the vowel list
            temp.update(ed_word)
    l.difference_update(temp)  # updating the set by removing words end with 'ed'
    return l


def syllable_count_filter(words, end_words:list):
    for end in end_words:                     # this function also work for filtering other ends of syllable
        pattern = "\w*[aeiou]\w*"
        if words is None:
            return []
        match = re.findall(pattern, words)  # getting list of words have vowels
        l = set(match)
        temp = set()
        for word in match:
            es_word = re.findall(f"\w+{end}$", word)
            if len(es_word) != 0:  # getting the words end with 'es' from the vowel list
                temp.update(es_word)
        l.difference_update(temp)  # updating the set by removing words end with 'es'
        temp.clear()
    return l


def positivity(words):
    for word in nltk.word_tokenize(words):
        if word in positive_corpus:
            p.add(word)
    score = len(p)
    p.clear()
    return score


def negativity(words):
    for word in nltk.word_tokenize(words):
        if word in negative_corpus:
            p.add(word)
    score = len(p)
    p.clear()
    return score


def polarity(l):
    polarity_scores = []
    for i in tqdm(l, desc="POLARITY SCORE", total=114):
        polarity_scores.append((float(i[0]) - float(i[1])) / ((float(i[0]) + float(i[1])) + 0.000001))
    return polarity_scores


def subjectivity(l):
    subjectivity_scores = []
    for i in tqdm(l, desc="SUBJECTIVITY SCORE", total=114):
        total_words = len(i[2])
        subjectivity_scores.append((float(i[0]) + float(i[1])) / (float(total_words) + 0.000001))
    return subjectivity_scores


def avg_sent_length(l):
    score = []
    for i in tqdm(l, desc="AVG SENTENCE LENGTH", total=114):
        score.append(float(i[0]) // float(i[1]))
    return score


def complex_word_count(words):
    pattern = "\w*[aeiou]\w*"
    if words is None:
        return {}
    match = re.findall(pattern, words)
    pattern2 = "[aeiou]"
    l = set()
    for word in match:
        n_vowels = re.findall(pattern2, word)
        if len(n_vowels) > 2:
            l.update(word)
    return l


def count_personal_pronouns(words):
    if words is None:
        return 0
    pattern = r'\b(I|we|my|ours|us|you|they|it|she|he)\b'
    matches = re.findall(pattern, words)
    if "US" in matches:
        matches.remove('US')
    return len(matches)


def average_word_length(sum_chars, sum_words):
    scores = []
    for i in tqdm(range(len(sum_chars)), desc="AVG WORD LENGTH", total=114):
        scores.append(float(sum_chars[i]) // float(sum_words[i]))
    return scores


def percentage_complex_words(complex_words, norm_words):
    scores = []
    for i in tqdm(range(len(complex_words)), desc="PERCENTAGE OF COMPLEX WORDS", total=114):
        scores.append((float(complex_words[i]) / float(norm_words[i]) * 100))
    return scores


def average_sentence_length(n_words, n_sentences):
    scores = []
    for i in tqdm(range(len(n_words)), desc="AVG SENTENCE LENGTH", total=114):
        scores.append(float(n_words[i]) // float(n_sentences[i]))
    return scores


def average_words_per_sentence(n_words, n_sentences):
    scores = []
    for i in tqdm(range(len(n_words)), desc="AVG NUMBER OF WORDS PER SENTENCE", total=114):
        scores.append(float(n_words[i]) // float(n_sentences[i]))
    return scores


def fog_index(avg_sent_lengths, complex_words):
    scores = []
    for i in tqdm(range(len(avg_sent_lengths)), desc="FOG INDEX", total=114):
        scores.append(0.4 * (avg_sent_lengths[i] + complex_words[i]))
    return scores


df_link = pd.read_excel('input.xlsx')
df = pd.DataFrame()

df['URL_ID'] = df_link['URL_ID']
df['URL'] = df_link['URL']

# df_link['url_text'] = df_link['URL'].apply(lambda x: extract(x))
df['url_text'] = extreme_Extraction(df_link['URL'])

print()
print("Wait few Minutes !!")
print()

os.makedirs('Text_files', exist_ok=True)
for url_id in df_link['URL_ID']:
    filepath = os.path.join(os.getcwd(), 'Text_files', '{}.txt'.format(str(int(url_id))))
    file = open(filepath, 'w', encoding='utf-8')
    text = df[df['URL_ID'] == url_id]['url_text']
    text = text.tolist()[0]
    file.write(str(text))

nltk.download('punkt')

print()
print("Wait few Minutes !!")
print()

df['num_characters'] = df['url_text'].apply(lambda x: len(str(x)))

df['num_words'] = df['url_text'].apply(lambda x: len(list(nltk.word_tokenize(str(x)))))

df['num_sentencecs'] = df['url_text'].apply(lambda x: len(list(nltk.sent_tokenize(str(x)))))

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

print()
print("Wait few Minutes !!")
print()

filepath = os.path.join(os.getcwd(), 'StopWords', 'StopWords_Auditor.txt')
StopWords_Auditor = open(filepath, 'r')
StopWords_Auditor = set(StopWords_Auditor.read().splitlines())
STOPWORDS = STOPWORDS.union(StopWords_Auditor)

filepath = os.path.join(os.getcwd(), 'StopWords', 'StopWords_DatesandNumbers.txt')
StopWords_DatesandNumbers = open(filepath, 'r')
StopWords_DatesandNumbers = set(StopWords_DatesandNumbers.read().splitlines())
l = filter(StopWords_DatesandNumbers)
STOPWORDS = STOPWORDS.union(set(l))

filepath = os.path.join(os.getcwd(), 'StopWords', 'StopWords_Currencies.txt')
StopWords_Currencies = open(filepath, 'r')
StopWords_Currencies = set(StopWords_Currencies.read().splitlines())
l = filter(StopWords_Currencies)

filepath = os.path.join(os.getcwd(), 'StopWords', 'StopWords_Generic.txt')
StopWords_Generic = open(filepath, 'r')
StopWords_Generic = set(StopWords_Generic.read().splitlines())
STOPWORDS = STOPWORDS.union(StopWords_Generic)

filepath = os.path.join(os.getcwd(), 'StopWords', 'StopWords_Geographic.txt')
StopWords_Geographic = open(filepath, 'r')
StopWords_Geographic = set(StopWords_Geographic.read().splitlines())
l = filter(StopWords_Geographic)
STOPWORDS = STOPWORDS.union(set(l))

filepath = os.path.join(os.getcwd(), 'StopWords', 'StopWords_Names.txt')
StopWords_Names = open(filepath, 'r')
StopWords_Names = set(StopWords_Names.read().splitlines())
l = filter(StopWords_Names)
STOPWORDS = STOPWORDS.union(set(l))

df['transformed_text'] = df['url_text'].apply(lambda x: transform_text(str(x)))

# df_out = pd.read_excel('Output_Data_Structure.xlsx')

filepath = os.path.join(os.getcwd(), 'MasterDictionary', 'negative-words.txt')
negative_file = open(filepath, 'r')
negative_corpus = set(negative_file.read().splitlines())

filepath = os.path.join(os.getcwd(), 'MasterDictionary', 'positive-words.txt')
positive_file = open(filepath, 'r')
positive_corpus = set(positive_file.read().splitlines())

df_link['POSITIVE SCORE'] = df['transformed_text'].apply(lambda x: positivity(x))

df_link['NEGATIVE SCORE'] = df['transformed_text'].apply(lambda x: negativity(x))

z = zip(df_link['POSITIVE SCORE'], df_link['NEGATIVE SCORE'])
df_link['POLARITY SCORE'] = polarity(z)

z = zip(df_link['POLARITY SCORE'], df_link['NEGATIVE SCORE'], df['transformed_text'])
df_link['SUBJECTIVITY SCORE'] = subjectivity(z)

df_link['AVG SENTENCE LENGTH'] = avg_sent_length(zip(df['num_words'], df['num_sentencecs']))


complex_word=df['url_text'].apply(lambda x: len(complex_word_count(x)))

average_sentence_length = average_sentence_length(df['num_words'], df['num_sentencecs'])




df_link['PERCENTAGE OF COMPLEX WORDS '] = percentage_complex_words(complex_word, df['num_words'])

df_link['FOG INDEX '] = fog_index(average_sentence_length, df_link['PERCENTAGE OF COMPLEX WORDS '])

df_link['AVG NUMBER OF WORDS PER SENTENCE'] = average_words_per_sentence(df['num_words'], df['num_sentencecs'])

df_link['COMPLEX WORD COUNT'] = complex_word

df_link['WORD COUNT'] = df['transformed_text'].apply(lambda x: len(nltk.word_tokenize(x)))

df_link['SYLLABLE PER WORD'] = df['url_text'].apply(lambda x: len(syllable_count(x)))

df_link['PERSONAL PRONOUNS'] = df['url_text'].apply(lambda x: count_personal_pronouns(x))

df_link['AVG WORD LENGTH'] = average_word_length(df['num_characters'], df['num_words'])


filepath = os.path.join(os.getcwd(), 'Output_Data_Structure.xlsx')
if os.path.exists(filepath):
    os.remove(filepath)

df_link.to_excel('Output_Data_Structure.xlsx')

print("Text_files Directory and Output_Data_Structure.xlsx created...")

print("Your Sentiment Analysis Successfully Completed")
print("Check File in Directory!!!")
