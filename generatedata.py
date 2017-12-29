import xml.etree.ElementTree
import nltk
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
e = xml.etree.ElementTree.parse('Laptops_Train_v2.xml').getroot()
with open("sentence_train.txt", "w") as text_file:
    for sent in e:
        print(sent[0].text.lstrip())
        stop = stopwords.words('english') + list(string.punctuation)
        tokens = [i for i in word_tokenize((sent[0].text.lstrip()).lower()) if i not in stop]
        print(nltk.pos_tag(tokens))
        print(tokens)
        print("\n")
        text_file.write(sent[0].text.lstrip())
        text_file.write("\n")
print("--------------------------------------------------------------------------------------------------------")        
with open("sentence_features_train.txt", "w") as text_file:
    for term in e.iter('aspectTerm'):
        print(term.attrib.get('term').lstrip())
        text_file.write(term.attrib.get('term').lstrip())
        text_file.write("\n")
print("--------------------------------------------------------------------------------------------------------")        

e = xml.etree.ElementTree.parse('Laptops_Test_Gold.xml').getroot()
with open("sentence_test.txt", "w") as text_file:
    for sent in e:
        print(sent[0].text.lstrip())
        text_file.write(sent[0].text.lstrip())
        text_file.write("\n")
print("--------------------------------------------------------------------------------------------------------") 
with open("sentence_features_test.txt", "w") as text_file:
    for term in e.iter('aspectTerm'):
        print(term.attrib.get('term').lstrip())
        text_file.write(term.attrib.get('term').lstrip())
        text_file.write("\n")
print("--------------------------------------------------------------------------------------------------------") 


