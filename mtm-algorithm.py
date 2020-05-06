#(1) Получить самые популярные русские слова на основе частоты

#(2) Создать контексты, переводя каждое слово в его семантический эквивалент на других языках с помощью онлайн-переводчика
# Каждое слово и его перевод является контекстом, и объединенные контексты для русских слов размером в 100 тысяч слов используются как embedding space (список контекстов)
# Чтобы повысить вероятность предсказания, повторить embedding space n-раз; n = 100

#(3) Обучить вложения с помощью модели word2vec CBOW
#размер окна (контекст) = 5, отрицательная выборка = 5, минимальное количество слов = 50, размер атрибутов = 3


#from googletrans import Translator
from translate import Translator
from gensim.test.utils import common_texts
import googletrans
translator=Translator(to_lang="en", from_lang='ru')
import  json,codecs,gensim,logging,re
from gensim.models import Word2Vec
from gensim import corpora
from gensim.utils import simple_preprocess
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#чтение слов
p=open('rus.txt', encoding='utf-8')
#top_100k_words = corpora.Dictionary(simple_preprocess(line, deacc=True) for line in open('wiki-100.txt', encoding='utf-8'))
top_100k_words=re.split(r'[;,?.–\s]\s*', p.read())


#создание embedding space
def embeddingspace(top_100k_words):
    embedding_space=[]
    for word in top_100k_words:
        translate_word=[translator.translate(word)]
        translate_word=[word]+translate_word
        print(translate_word)
        embedding_space.append(translate_word)
#        print(embedding_space)
    return sum([embedding_space]*100,[])
        

#обучение модели word2vec с embedding space
def genmodel(top_100k_words):
    embedding=embeddingspace(top_100k_words)
    print('*******************************************************************')
    model=Word2Vec(embedding, window=5, min_count=50, size=300, sg=1, workers=3)
    model.save("c:\\diplom\\word2vec.model")
    print(model.wv.most_similar('друг'))
    return


if __name__ == "__main__":
    trained_mtm_model=genmodel(top_100k_words)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
 #   print(trained_mtm_model)
 #   test_model=trained_mtm_model.most_similar('human')
 #   print(test_model)



