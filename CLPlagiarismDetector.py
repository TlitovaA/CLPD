#(1) Получить самые популярные русские слова на основе частоты

#(2) Создать контексты, переводя каждое слово в его семантический эквивалент на других языках с помощью онлайн-переводчика
# Каждое слово и его перевод является контекстом, и объединенные контексты для русских слов размером в 100 тысяч слов используются как embedding space (список контекстов)
# Чтобы повысить вероятность предсказания, повторить embedding space n-раз; n = 100

#(3) Обучить вложения с помощью модели word2vec CBOW
#window size (context) =5, negative sampling=5,minimum word count=50, attributes size =3
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
import os

nltk.download('wordnet')

morph=pymorphy2.MorphAnalyzer()
nlp = spacy.load('en_core_web_sm')


#создание embeddingspace
def embeddingspace():
    embedding_space=[]
    f1 = open('rusDictionary.txt', encoding='utf-8')
    f2 = open('engDictionary.txt', encoding='utf-8')
    f3 = open('engSynDictionary.txt', encoding='utf-8')    
    line1 = f1.readline()
    line2 = f2.readline()
    line3 = f3.readline()
    while line1 and line2 and line3:
        line1 = f1.readline().lower().splitlines()
        line2 = f2.readline().lower().splitlines()
        line3 = f3.readline().lower().splitlines()
        line4 = line1+line2+line3
        embedding_space.append(line4)
    f1.close()
    f2.close()
    f3.close()
    return sum([embedding_space]*100,[])

english_stopwords_extention=['always','ever','anyone','else','never','however', 'although','except',
                             'though','nevertheless','th','st','nd','d','rd','could','unless',
                             'mrs','mr','nor','appears','also','this','either','prospective',
                             'almost','originally','several','since','usually','usual','various','ensure',
                             'would','till','upon','exact','able','wherever','iii','obtain',
                             'fulfil','\ufeffthe','perform','dun','enfant']


#чтение документа и разбиение на токены
def text_reader_tokenizer(filename, language):
    stop_words = set(stopwords.words(language)) 
    p=open(filename, encoding='utf-8')
    splitted_text=re.split(r'[;,-?»«–\s]\s*', p.read())
    tokens_without_sw = [word for word in splitted_text if not word in stop_words] 
    words_in_sentences=[]
    for word in tokens_without_sw:
        if (word!=''):
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
    reg='\/|\;|\:|\,|\-|\"|\'|\(|\)|\=|\+|\–|\«|\»|\^|\%|\$|\@|\*|\~|\[|\]|\{|\}|\&|\<|\>|\>|\ˆ|\∈|\`|\_|\№|\#|[0-9]+|\|'
    if(language=='russian'):
        for sent in doc:
            sent = re.sub(reg, '', sent)
            if(len(sent)>15):
                for word1 in sent.split():
                    w1=morph.parse(word1)[0].normal_form
                    if w1 not in get_stop_words(language) and len(w1)>2:
                        sentarr=sentarr+str(w1)+' '
                        w1=''
                if(sentarr!='' and len(sentarr)>15):
                    sentences.append(sentarr)
                    sentarr=''
    #    print(sentences)                
        return sentences

    if(language=='english'):
        stop_words = stopwords.words(language)
        stop_words.extend(english_stopwords_extention)
        for sent in doc:
            sent = re.sub(reg, '', sent)
            if(len(sent)>15):
                for word1 in sent.split():
                    w1=nlp(word1)
                    for token in w1:
                        if (not token.lemma_ in stop_words and token.lemma_!='-PRON-' and len(token.lemma_)>2):
                            sentarr=sentarr+str(token.lemma_)+' '
                            w1=''
                if(sentarr!='' and len(sentarr)>15):
                    sentences.append(sentarr)
                    sentarr=''
     #   print(sentences)
        return sentences
     

#обучение модели Word2Vec
def genmodel():
    embedding=embeddingspace()
    model=Word2Vec(embedding, window=3, negative=10, size=3, sg=1, workers=5)
    model.save('word2vec.model')
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
    stop_words.extend(english_stopwords_extention)
    lemmatized_text=[]
    for word1 in doc:
        w1=nlp(word1)
        for token in w1:
            if (token.lemma_!='-PRON-'):
                lemmatized_text.append(token.lemma_)
    tokens_without_sw = [word for word in lemmatized_text if not word in stop_words] 
    return tokens_without_sw


#параметризовать модель (загрузка,обучение)
if __name__ == "__main__":
#    trained_mtm_model=genmodel()
    print(os.path)
    trained_mtm_model = Word2Vec.load('word2vec.model')

    '''
    print(trained_mtm_model.wv.similarity('sun','солнце'))
    print(trained_mtm_model.wv.similarity('song','песня'))
    print(trained_mtm_model.wv.most_similar('слово'))
    '''
    i=1
    while(i<2):
        part_of_plagiarism=0
        russentences_count=0

        russian_doc_name='docs/author X/rus'+str(i)+'.txt'
     #   russian_doc_name='docs/source-document00007.txt'
        if(i<10):
            english_doc_name='docs/author X/en.rus'+str(i)+'.txt'
     #       english_doc_name='docs/suspicious-document0000'+str(i)+'.txt'
        else:
            english_doc_name='docs/author X/en'+str(i)+'.txt'
       #     english_doc_name='docs/suspicious-document000'+str(i)+'.txt'

    #    russian_doc_name='rustext.txt'
    #    english_doc_name='engtext.txt'

     
        russian_language='russian'
        english_language='english'

        rusdoc=text_reader_tokenizer(russian_doc_name,russian_language)
        engdoc=text_reader_tokenizer(english_doc_name,english_language)
        lemmatized_rusdoc=russian_lemmatizer(rusdoc)
        lemmatized_engdoc=english_lemmatizer(engdoc)

        russentences=text_reader_bydots_with_preprocessing(russian_doc_name,russian_language)
        russentences_count=len(russentences)

        engsentences=text_reader_bydots_with_preprocessing(english_doc_name,english_language)
        engsentences_count=len(engsentences)

        list_of_stopw=[]
        count=0
        for russent in russentences:
            p=0
            ru=russent.split()
            for engsent in engsentences:
                en=engsent.split()
                try:
                    distance=trained_mtm_model.wv.n_similarity(ru, en)
                    if (distance>0.99):
                        print('['+str(count+1)+']'+str(russent)+' '+str(engsent)+' '+str(distance))
                        count=count+1
                        p=1
                        if (p==1):
                            break
                except KeyError as e:
                  #  print ('I got a KeyError - reason %s' % str(e))               
                  #  pass
                    s = str(e)
                    pattern = 'word '(.*?)' not in vocabulary'
                    substring = re.search(pattern, s).group(1)
                    try:
                        en.remove(substring)
                        list_of_stopw.append(substring)
                        try:
                            distance=trained_mtm_model.wv.n_similarity(ru, en)
                            if (distance>0.99):
                                print("["+str(count+1)+"]"+str(russent)+" "+str(engsent)+" "+str(distance))
                                count=count+1
                                p=1
                                if (p==1):
                                    break
                        except KeyError as e:
                            e=e
                        except ZeroDivisionError as zd:
                            e=e
                            break
                        
                    except ValueError as e:
                        try:
                            ru.remove(substring)
                            try:
                                distance=trained_mtm_model.wv.n_similarity(ru, en)
                                if (distance>0.99):
                                    print("["+str(count+1)+"]"+str(russent)+" "+str(engsent)+" "+str(distance))
                                    count=count+1
                                    p=1
                                    if (p==1):
                                        break
                            except KeyError as e:
                                e=e
                            except ZeroDivisionError as zd:
                                e=e
                        except ValueError as e:
                            e=e
                            break
                except ZeroDivisionError as zd:
                    zd=zd
                    break
                except UnicodeEncodeError as uer:
                    uer=uer


        part_of_plagiarism=count/russentences_count
        if(part_of_plagiarism>0.7):
            d=open('RESULT_'+str(i)+' + '+str(part_of_plagiarism)+'.txt','w',encoding='utf-8')
            d.write('*************  PLAGIARISED  *************' +'\n'+'\n')
        else:
            d=open('RESULT_'+str(i)+'.txt', 'w', encoding='utf-8')
            
        d.write('*** '+russian_doc_name+'\n')
        d.write('Sentences: '+str(russentences_count)+'\n'+'\n')
        d.write('*** '+english_doc_name+'\n')
        d.write('Sentences: '+str(engsentences_count)+'\n'+'\n')

        d.write('Count of plagiarised sentences: '+str(count)+'\n')
        d.write('Part Of Plagiarism: '+str(part_of_plagiarism)+'\n')
        
   #     print(list_of_stopw)
        i=i+1
        d.close()

