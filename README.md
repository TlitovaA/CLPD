#SynB is a cross-lingual plagiarism detection tool in Russian-English language pair.

##Required Python 3.7

* CLPlagoarismDetector.py - the main module in which the Word2vec model is trained and articles are analyzed.
* dictionaryTranslator.py - a module for translating a Russian-language dictionary into its equivalent in English.
* engParaphraser.py - a module for compiling a dictionary of synonyms for an English-language dictionary.

##The following Python libraries and modules are used:
* logging
* re
* stop_words
* pymorphy2
* nltk
* spacy
* googletrans

##Dictionaries:

* rusDictionary - file contains 90 193 Russian words. It is created from ProLing dictionary after lemmatization, deleting stop-words and duplicates that appeared after lemmatization.
* engSynDictionary - file contains 90 193 English equivalents for Russian words. It is created after running module *dictionaryTranslator.py*. So it's not neccessary to run this module again, if you have this dictionary.
* engDictionary - file contains 90 193 English synonyms for English words. It is created after running module *engParaphraser.py*. So it's not neccessary to run this module again, if you have this dictionary.

##Usage

Provide the names of the text files you want to compare in module CLPlagiarismDetector (sourse document - **russian_doc_name**, suspicious document - **english_doc_name**) and run it. If you want analyze several suspicious documents then change number of suspicious files in **number_of_suspicious_docs**. Results will be available in .txt file in your directory.