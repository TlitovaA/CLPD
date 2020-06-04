import logging, re
from gensim.models import Word2Vec
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
from stop_words import get_stop_words
from nltk.corpus import stopwords
import pymorphy2
import nltk
import spacy

nltk.download('wordnet')

morph = pymorphy2.MorphAnalyzer()
nlp = spacy.load('en_core_web_sm')


#создание embeddingspace
def embeddingspace():
    embedding_space = []
    f1 = open('rusDictionary.txt', encoding = 'utf-8')
    f2 = open('engDictionary.txt', encoding = 'utf-8')
    f3 = open('engSynDictionary.txt', encoding = 'utf-8')    
    line1 = f1.readline()
    line2 = f2.readline()
    line3 = f3.readline()
    while line1 and line2 and line3:
        line1 = f1.readline().lower().splitlines()
        line2 = f2.readline().lower().splitlines()
        line3 = f3.readline().lower().splitlines()
        line4 = line1 + line2 + line3
        embedding_space.append(line4)
    f1.close()
    f2.close()
    f3.close()
    return sum([embedding_space] * 50, [])

english_stopwords_extention = ['always','ever','anyone','else','never','however', 'although','except',
                             'though','nevertheless','th','st','nd','d','rd','could','unless',
                             'mrs','mr','nor','appears','also','this','either',
                             'almost','originally','several','since'
                             'would','till','upon','exact','able','wherever','iii','\ufeffthe']


#чтение документа и разбиение на токены
def text_reader_tokenizer(filename, language):
    stop_words = set(stopwords.words(language)) 
    p = open(filename, encoding = 'utf-8')
    splitted_text = re.split(r'[;,-?»«–\s]\s*', p.read())
    tokens_without_sw = [word for word in splitted_text if not word in stop_words] 
    words_in_sentences = []
    for word in tokens_without_sw:
        if (word != ''):
            words_in_sentences.append(word.lower())
    return words_in_sentences


#чтение документа по предложениям
def text_reader_bydots_with_preprocessing(filename, language):
    p = open(filename, encoding = 'utf-8')
    line = p.read().lower()
    doc = re.split(r'[.?!]\s*', line)
    sentences = []
    sentarr = ''
    reg = '\/|\;|\:|\,|\-|\"|\'|\(|\)|\=|\+|\–|\«|\»|\^|\%|\$|\@|\*|\~|\[|\]|\{|\}|\&|\<|\>|\>|\ˆ|\∈|\`|\_|\№|\#|[0-9]+|\|'
    if(language == 'russian'):
        for sent in doc:
            sent = re.sub(reg, '', sent)
            if(len(sent) > 15):
                for word1 in sent.split():
                    w1 = morph.parse(word1)[0].normal_form
                    if w1 not in get_stop_words(language) and len(w1) > 2:
                        sentarr = sentarr+str(w1) + ' '
                        w1 = ''
                if(sentarr != '' and len(sentarr) > 15):
                    sentences.append(sentarr)
                    sentarr = ''
        return sentences

    if(language == 'english'):
        stop_words = stopwords.words(language)
        stop_words.extend(english_stopwords_extention)
        for sent in doc:
            sent = re.sub(reg, '', sent)
            if(len(sent) > 15):
                for word1 in sent.split():
                    w1 = nlp(word1)
                    for token in w1:
                        if (not token.lemma_ in stop_words and token.lemma_ != '-PRON-' and len(token.lemma_) > 2):
                            sentarr = sentarr + str(token.lemma_) + ' '
                            w1=''
                if(sentarr != '' and len(sentarr) > 15):
                    sentences.append(sentarr)
                    sentarr = ''
        return sentences
     

#обучение модели Word2Vec
def genmodel():
    embedding = embeddingspace()
    model = Word2Vec(embedding, window = 3, negative = 3, size = 3, sg = 1, workers = 5)
    model.save('word2vec.model')
    return model


#лемматизатор русскоязычного документа
def russian_lemmatizer(doc):
    lemmatized_text = []
    for word1 in doc:
        w1 = morph.parse(word1)[0].normal_form
        lemmatized_text.append(w1)
    tokens_without_sw = [word for word in lemmatized_text if word not in get_stop_words('russian')]
    return tokens_without_sw


#лемматизатор англоязычного документа
def english_lemmatizer(doc):
    stop_words = stopwords.words('english')
    stop_words.extend(english_stopwords_extention)
    lemmatized_text = []
    for word1 in doc:
        w1 = nlp(word1)
        for token in w1:
            if (token.lemma_ != '-PRON-'):
                lemmatized_text.append(token.lemma_)
    tokens_without_sw = [word for word in lemmatized_text if not word in stop_words] 
    return tokens_without_sw


if __name__ == "__main__":
    try:
        trained_mtm_model = Word2Vec.load('word2vec.model')
        print('Модель загружена. Идёт анализ текста...')
    except FileNotFoundError as fnf:
        trained_mtm_model = genmodel()
        print('Модель обучена. Идёт анализ текста...')

    i = 1
    number_of_suspicious_docs = 8
    while(i < number_of_suspicious_docs):
        part_of_plagiarism = 0
        russentences_count = 0
 
        russian_doc_name = 'docs/source-document00001.txt'
        english_doc_name = 'docs/suspicious-document0000' + str(i) + '.txt'
       
        russian_language = 'russian'
        english_language = 'english'

        rusdoc = text_reader_tokenizer(russian_doc_name, russian_language)
        engdoc = text_reader_tokenizer(english_doc_name, english_language)
        lemmatized_rusdoc = russian_lemmatizer(rusdoc)
        lemmatized_engdoc = english_lemmatizer(engdoc)

        russentences = text_reader_bydots_with_preprocessing(russian_doc_name, russian_language)
        russentences_count = len(russentences)

        engsentences = text_reader_bydots_with_preprocessing(english_doc_name, english_language)
        engsentences_count = len(engsentences)

        list_of_stopw = []
        count = 0
        for russent in russentences:
            if(count > (engsentences_count - 1)):
                break
            p = 0
            ru = russent.split()
            for engsent in engsentences:
                en = engsent.split()
                try:
                    distance = trained_mtm_model.wv.n_similarity(ru, en)
                    if (distance > 0.97):
                        count = count + 1
                        p = 1
                        if (p == 1):
                            break
                except KeyError as e:
                    s = str(e)
                    pattern = "word '(.*?)' not in vocabulary"
                    substring = re.search(pattern, s).group(1)
                    try:
                        en.remove(substring)
                        list_of_stopw.append(substring)
                        try:
                            distance = trained_mtm_model.wv.n_similarity(ru, en)
                            if (distance > 0.97):
                                count = count + 1
                                p = 1
                                if (p == 1):
                                    break
                        except KeyError as e:
                            e = e
                        except ZeroDivisionError as zd:
                            e = e
                            break
                        
                    except ValueError as e:
                        try:
                            ru.remove(substring)
                            try:
                                distance = trained_mtm_model.wv.n_similarity(ru, en)
                                if (distance > 0.97):
                                    count = count + 1
                                    p = 1
                                    if (p == 1):
                                        break
                            except KeyError as e:
                                e = e
                            except ZeroDivisionError as zd:
                                e = e
                                break
                        except ValueError as e:
                            e = e
                except ZeroDivisionError as zd:
                    zd = zd
                    break
                except UnicodeEncodeError as uer:
                    uer = uer


        part_of_plagiarism = count / russentences_count
        if(part_of_plagiarism > 0.8):
            d = open('RESULT_' + str(i) + ' + ' + str(part_of_plagiarism) + '.txt', 'w',encoding = 'utf-8')
            d.write('*************  ПЛАГИАТ  *************' + '\n' + '\n')
        else:
            d = open('RESULT_'+str(i) + '.txt', 'w', encoding = 'utf-8')
            
        d.write('Источник: ' + russian_doc_name + '\n')
        d.write('Количество предложений: '+str(russentences_count) + '\n' + '\n')
        d.write('Источник подозреваемого документа: ' + english_doc_name + '\n')
        d.write('Количество предложений: ' + str(engsentences_count) + '\n' + '\n')

        d.write('Количество заимствованных предложений: ' + str(count) + '\n')
        d.write('Доля плагиата: ' + str(part_of_plagiarism) + '\n')
        
        i = i + 1
        d.close()
        print("Проверка прошла успешно!")
        input('Нажмите ENTER') 

