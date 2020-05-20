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
def text_reader_bydots(filename):
    p=open(filename, encoding='utf-8')
    line = p.read().lower()
    print(line)
    doc = re.split(r'[.?]\s*', line)
    sentences=[]
    for sent in doc:
        if(len(sent)>2):
            sent = re.sub('[,–«»:;()]', '', sent)
            sentences.append(sent.lower())
            print(sent)
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


#лемматизатор русскоязычных предложений
def russiansentense_lemmatizer(sent):
    lemmatized_sent=[]
    for word1 in sent.split():
        w1=morph.parse(word1)[0].normal_form
        lemmatized_sent.append(w1)
    tokens_without_sw = [word for word in lemmatized_sent if word not in get_stop_words('russian')]
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


#лемматизатор англоязычных предложений
def englishsentence_lemmatizer(sent):
    stop_words = stopwords.words('english')
    stop_words.extend(['anyone', 'else', '-'])
    lemmatized_text=[]
    for word1 in sent.split():
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

    rusdoc=text_reader_tokenizer('rusdoc.txt','russian')
    engdoc=text_reader_tokenizer('engdoc.txt','english')
    lemmatized_rusdoc=russian_lemmatizer(rusdoc)
    lemmatized_engdoc=english_lemmatizer(engdoc)

    print(lemmatized_rusdoc)
    print(lemmatized_engdoc)

    try:
        distance = trained_mtm_model.wv.n_similarity(lemmatized_rusdoc, lemmatized_engdoc)
        print(distance)
    except KeyError as e:
        print ('I got a KeyError - reason "%s"' % str(e))
        pass

    russentences = text_reader_bydots('rusdoc.txt')
    engsentences = text_reader_bydots('engdoc.txt')

    for russent in russentences:
        rusvector=russiansentense_lemmatizer(russent)
        for engsent in engsentences:
            engvector=englishsentence_lemmatizer(engsent)
            try:
                distance = trained_mtm_model.wv.n_similarity(rusvector, engvector)
                if(distance>0.88):
                    print(str(rusvector)+" "+str(engvector)+" "+str(distance))
            except KeyError as e:
                # print ('I got a KeyError - reason %s' % str(e))               
                pass



