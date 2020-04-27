#Взять топ-n самых распространённых русских слов

#Создать список контекстов (embedding space), реплицировать n-раз (100)

#Обучить вложения (embeddings) с помощью модели word2vec CBOW
#window size (context)=5, negative sampling=5, minimum word count=50, attributes size=3

#INSTALL:
#google translate: pip install googletrans
#Gensim: pip install gensim


from googletrans import Translator
translator=Translator() 
import  json,codecs,gensim,logging
from gensim.models import Word2Vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#код для языка в соответствии с Google Translate
languages=['rus']


#создание embedding space
def embeddingspace(top_100k_words):
    embedding_space=[]
    for word in top_100k_words:
        translate_word=[translator.translate(word,lang) for lang in languages]
        translate_word=[word]+translate_word
        embedding_space.append(translate_word)
    return sum([embedding_space]*100,[])
        

#обучение модели word2vec с embedding space
def genmodel(top_100k_words,n1,n2,n3):
   embedding=embeddingspace(top_100k_words)
   model=gensim.models.Word2Vec(lis1,window=param1, min_count=param2, size=300, sg=1, workers=param3)
   model.save("c:\\model_name")
return

if __name__ == "__main__":
    trained_mtm_model=genemodel(100k_Eng_words,param1,param2,param3)
    #тестирование модели любым словом ('друг', 'friend')
    test_model=trained_mtm_model.most_similar('друг')
    print(test_model)

