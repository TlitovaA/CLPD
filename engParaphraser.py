from nltk.corpus import wordnet
import re

p=open('engDictionary.txt', encoding='utf-8')
d=open('engSynDictionary.txt','w',encoding='utf-8')

y=''
splitted_text=re.split(r'[\r\n]+', p.read())

i=0
while(i<len(splitted_text)):

    s=splitted_text[i]
    x=''

    synonyms = []
    for syn in wordnet.synsets(s):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())

    if(len(synonyms) != 0):
        for word in synonyms:
            if "_" not in word and s not in word and word!='':
                x=word.lower()
                d.write(x+'\n')
                break
        if (x==''):
            d.write(s+'\n')
    else:
        d.write(s+'\n')
    
    i=i+1
       
d.close()
print('Done')
