#(1) Получить самые популярные русские слова на основе частоты

#(2) Создать контексты, переводя каждое слово в его семантический эквивалент на других языках с помощью онлайн-переводчика
# Каждое слово и его перевод является контекстом, и объединенные контексты для русских слов размером в 100 тысяч слов используются как embedding space (список контекстов)
# Чтобы повысить вероятность предсказания, повторить embedding space n-раз; n = 100

#(3) Обучить вложения с помощью модели word2vec CBOW
#размер окна (контекст) = 5, отрицательная выборка = 5, минимальное количество слов = 50, размер атрибутов = 3

from gensim.test.utils import common_texts
import json,codecs,gensim,logging,re
from gensim.models import Word2Vec
from gensim import corpora
from gensim.utils import simple_preprocess
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from stop_words import get_stop_words
from nltk.corpus import stopwords
import pymorphy2
from nltk.stem import WordNetLemmatizer
import nltk
import spacy

nltk.download('wordnet')

morph=pymorphy2.MorphAnalyzer()
nlp = spacy.load('en_core_web_sm')

#создание embeddingspace
def embeddingspace():
    embedding_space=[]
    f1 = open('lemdic.txt', encoding='utf-8')
    f2 = open('englemdic.txt', encoding='utf-8')
    line1 = f1.readline()
    line2 = f2.readline()
    while line1 and line2:
        line1 = f1.readline().lower().splitlines()
        line2 = f2.readline().lower().splitlines()
        line3 = line1+line2
        embedding_space.append(line3)
    f1.close()
    f2.close()
    return sum([embedding_space]*100,[])


#чтение документа и разбиение на токены
def text_reader_tokenizer(filename, language):
    stop_words = set(stopwords.words(language)) 
    p=open(filename, encoding='utf-8')
    #top_100k_words = corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open('wiki-100.txt', encoding='utf-8'))
    splitted_text=re.split(r'[;,-?»«–\s]\s*', p.read())
    tokens_without_sw = [word for word in splitted_text if not word in stop_words] 
    words_in_sentences=[]
    for word in tokens_without_sw:
        if (word!=""):
            words_in_sentences.append(word.lower())
    return words_in_sentences


#чтение документа по предложениям
def text_reader_bydots_with_preprocessing(filename, language):
    p=open(filename, encoding='utf-8')
    line = p.read().lower()
 #   print(line)
    doc = re.split(r'[.?!]\s*', line)
    sentences=[]
    sentarr=''
    if(language=='russian'):
        for sent in doc:
            sent = re.sub('\/|\;|\:|\,|\-|\"|\'|\(|\)|\=|\+|\–|\«|\»|[0-9]+', '', sent)
            if(len(sent)>10):
                for word1 in sent.split():
                    w1=morph.parse(word1)[0].normal_form
                    if w1 not in get_stop_words(language) and len(w1)>2:
                        sentarr=sentarr+str(w1)+' '
                        w1=''
                if(sentarr!=''):
                    print(sentarr)
                    sentences.append(sentarr)
                    sentarr=''
        print(sentences)                
        return sentences

    if(language=='english'):
        stop_words = stopwords.words(language)
        stop_words.extend(['anyone', 'else', 'never', 'however', 'although', 'though', 'nevertheless', 'th', 'st', 'nd', 'd', 'rd', 'mrs', 'mr'])
        for sent in doc:
            sent = re.sub('\/|\;|\:|\,|\-|\"|\'|\(|\)|\=|\+|\–|\«|\»|[0-9]+', '', sent)
            if(len(sent)>10):
                for word1 in sent.split():
                    w1=nlp(word1)
                    for token in w1:
                        if (token.lemma_!='-PRON-' and not token.lemma_ in stop_words and len(token.lemma_)>2):
                            sentarr=sentarr+str(token.lemma_)+' '
                            w1=''
                if(sentarr!=''):
                    print(sentarr)
                    sentences.append(sentarr)
                    sentarr=''
        print(sentences)
        return sentences

      

#обучение модели Word2Vec
def genmodel():
    embedding=embeddingspace()
    model=Word2Vec(embedding, window=5, min_count=50, size=300, sg=1, workers=3)
    model.save("c:\\diplom\\word2vec.model")
    return model


#лемматизатор русскоязычного документа
def russian_lemmatizer(doc):
    lemmatized_text=[]
    for word1 in doc:
        w1=morph.parse(word1)[0].normal_form
        lemmatized_text.append(w1)
    tokens_without_sw = [word for word in lemmatized_text if word not in get_stop_words('russian')]
    return tokens_without_sw


#лемматизатор англоязычного документа
def english_lemmatizer(doc):
    stop_words = stopwords.words('english')
    stop_words.extend(['anyone', 'else'])
    lemmatized_text=[]
    for word1 in doc:
        w1=nlp(word1)
        for token in w1:
            if (token.lemma_!='-PRON-'):
                lemmatized_text.append(token.lemma_)
    tokens_without_sw = [word for word in lemmatized_text if not word in stop_words] 
    return tokens_without_sw



if __name__ == "__main__":
#    trained_mtm_model=genmodel()
    trained_mtm_model = Word2Vec.load("c:\\diplom\\word2vec.model")
    
    print(trained_mtm_model.wv.similarity("вера","absolutely"))
    print(trained_mtm_model.wv.most_similar("absolutely"))

    russian_doc_name="rusdoc.txt"
    english_doc_name="source-document00001.txt"
    russian_language='russian'
    english_language='english'

    rusdoc=text_reader_tokenizer(russian_doc_name,'russian')
    engdoc=text_reader_tokenizer(english_doc_name,'english')
    lemmatized_rusdoc=russian_lemmatizer(rusdoc)
    lemmatized_engdoc=english_lemmatizer(engdoc)

  #  print(lemmatized_rusdoc)
  #  print(lemmatized_engdoc)

    russentences = text_reader_bydots_with_preprocessing(russian_doc_name,russian_language)
    engsentences = text_reader_bydots_with_preprocessing(english_doc_name,english_language)

    for russent in russentences:
        ru=russent.split()
        for engsent in engsentences:
            en=engsent.split()
            try:
                distance=trained_mtm_model.wv.n_similarity(ru, en)
                if (distance>0.70):
                    print(str(russent)+" "+str(engsent)+" "+str(distance))
            except KeyError as e:
              #  print ('I got a KeyError - reason %s' % str(e))               
                pass



