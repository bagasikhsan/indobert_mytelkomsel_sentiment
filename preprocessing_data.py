import re
from nltk.tokenize import RegexpTokenizer
# Mengubah kalimat ke kata dasar menggunakan sastrawi
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

# load text files
txt_stopword = pd.read_csv(f"data/stopword_mytelkom.txt", names = ["stopwords"], header= None)
stopword_full = set(txt_stopword.stopwords.values)
normalizad_word = pd.read_csv(f"data/normalisasi.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

regexp = RegexpTokenizer('\w+')


def cleaning(text):
    # Remove URLs (https/http) from review_text
    text = re.sub(r'https?\S+', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'#\S+', ' ', text)
    text = re.sub(r'\'\w+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    return text.strip()  # Remove leading/trailing whitespace


# remove stopword 
def stopwords_removal(words):
    return [word for word in words if word not in stopword_full]

# Normalize words
def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]


def preprocessing_mytelkom(review):
  result = review.lower()   # mengubah teks ke huruf kecil
  result = cleaning(result)  
  result = regexp.tokenize(result)  # memisahkan kata-kata
  result = stopwords_removal(result)
  result = [stemmer.stem(word) for word in result] # Stemming
  result = normalized_term(result)
  result = " ".join(word for word in result)

  return result
