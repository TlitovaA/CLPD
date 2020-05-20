#Модуль для составления словаря, по которому обучается Word2Vec

#Скачен словарь ProLing на русском языке с сайта http://www.speakrus.ru/dict/ 

#Лемматизирован, очищен от стоп-слов и переведён на английский с помощью Google translate

from googletrans import Translator
translator=Translator()
import googletrans
import  json,codecs,gensim,logging,re
from gensim.models import Word2Vec
from gensim import corpora
from gensim.utils import simple_preprocess
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from stop_words import get_stop_words
import pymorphy2
from nltk.stem import WordNetLemmatizer
import nltk

morph=pymorphy2.MorphAnalyzer()
lemmatizer = WordNetLemmatizer()

languages=['en']

#чтание и разбиение на токены
def text_reader_tokenizer(filename, language):
    p=open(filename, encoding='utf-8')
    splitted_text=re.split(r'[;,?»«.\s]\s*', p.read())
    tokens_without_sw = [word for word in splitted_text if word not in get_stop_words(language)]
    words_in_sentences=[]
    for word in tokens_without_sw:
        if (word!=""):
            words_in_sentences.append(word.lower())
    return words_in_sentences

rusdoc=text_reader_tokenizer('proling_russian_dictionary.txt','russian')

f=open('lemmatized_russian_dictionary.txt', 'w', encoding='utf-8')
lemmatized_russian_dictionary=[]
for word1 in rusdoc:  
    w1=morph.parse(word1)[0].normal_form
    lemmatized_russian_dictionary.append(w1)

lemmatized_russian_dictionary=list(dict.fromkeys(lemmatized_russian_dictionary))

for word1 in lemmatized_russian_dictionary:      
    f.write(str(word1) + '\n')

f.close()


p=open('lemmatized_russian_dictionary.txt', encoding='utf-8')
f1=open('english_dictionary.txt', 'w', encoding='utf-8')
splitted_text=re.split(r'[;,?»«.\s]\s*', p.read())

for word in splitted_text:
    translated_word=[translator.translate(word,lang).text for lang in languages]
    w=translated_word[0].lower()
    print(str(w))
    f1.write(str(w) + '\n')
    
f1.close()
print('Done')
