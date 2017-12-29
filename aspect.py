#!/usr/bin/python

import sys
import cPickle as pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import codecs
import nltk
from nltk.corpus import treebank
import re
from nltk.corpus import stopwords
import string

def train_ner(pickle_file):
  # initialize
  pos_tagger = train_pos_tagger()
  ceps = ce_phrases()
  cep_words = ce_phrase_words(ceps)
  
  pickle_file = "NaiveBayesClassifierce_ner_classifier.pkl"

  sentfile = codecs.open("sentence_train.txt", 'rb','utf-8')
  featuresets = []
  for sent in sentfile:
    tagged_sent = tag(sent, pos_tagger, ceps, cep_words)
    for idx, (word, pos_tag, io_tag) in enumerate(tagged_sent):
      featuresets.append((word_features(tagged_sent, idx), io_tag))
  sentfile.close()

  """
   train classifier
   
  """
  classifier = nltk.NaiveBayesClassifier.train(featuresets)
  """
   evaluate classifier
   
  """
  
  if pickle_file != None:
    pickled_classifier = open(pickle_file, 'wb')
    pickle.dump(classifier, pickled_classifier)
    pickled_classifier.close()
  """
   test classifier
   
  """

  ceps_test = ce_phrases_test()
  cep_words_test = ce_phrase_words(ceps_test)
  sentfile_test = codecs.open("sentence_test.txt", 'rb','utf-8')
  featuresets_test = []
  for sent_test in sentfile_test:
    tagged_sent_test = tag(sent_test, pos_tagger, ceps_test, cep_words_test)
    for idx, (word, pos_tag, io_tag) in enumerate(tagged_sent_test):
      featuresets_test.append((word_features(tagged_sent_test, idx), io_tag))
  sentfile_test.close()
  print "accuracy NaiveBayesClassifier=", nltk.classify.accuracy(classifier, featuresets_test)
  
  pickle_file = "DecisionTreeClassifierce_ner_classifier.pkl"
  
  """
   train classifier
   
  """
  classifier = nltk.DecisionTreeClassifier.train(featuresets)
  """
   evaluate classifier
   
  """
  
  if pickle_file != None:
    pickled_classifier = open(pickle_file, 'wb')
    pickle.dump(classifier, pickled_classifier)
    pickled_classifier.close()
  """
   test classifier
   
  """
  ceps_test = ce_phrases_test()
  cep_words_test = ce_phrase_words(ceps_test)
  sentfile_test = codecs.open("sentence_test.txt", 'rb','utf-8')
  featuresets_test = []
  for sent_test in sentfile_test:
    tagged_sent_test = tag(sent_test, pos_tagger, ceps_test, cep_words_test)
    for idx, (word, pos_tag, io_tag) in enumerate(tagged_sent_test):
      featuresets_test.append((word_features(tagged_sent_test, idx), io_tag))
  sentfile_test.close()
  print "accuracy DecisionTreeClassifier=", nltk.classify.accuracy(classifier, featuresets_test)  
  
  pickle_file = "MaxentClassifierce_ner_classifier.pkl"
  """
   train classifier
   
  """
  classifier = nltk.MaxentClassifier.train(featuresets, algorithm="GIS", trace=0)

  """
   evaluate classifier
   
  """
    
  if pickle_file != None:
    pickled_classifier = open(pickle_file, 'wb')
    pickle.dump(classifier, pickled_classifier)
    pickled_classifier.close()
  """
   test classifier
   
  """

  ceps_test = ce_phrases_test()
  cep_words_test = ce_phrase_words(ceps_test)
  sentfile_test = codecs.open("sentence_test.txt", 'rb','utf-8')
  featuresets_test = []
  for sent_test in sentfile_test:
    tagged_sent_test = tag(sent_test, pos_tagger, ceps_test, cep_words_test)
    for idx, (word, pos_tag, io_tag) in enumerate(tagged_sent_test):
      featuresets_test.append((word_features(tagged_sent_test, idx), io_tag))
  sentfile_test.close()
  print "accuracy MaxentClassifier=", nltk.classify.accuracy(classifier, featuresets_test)
  
  
  
  return classifier

def get_trained_ner(pickle_file):
  pickled_classifier = open(pickle_file, 'rb')
  classifier = pickle.load(pickled_classifier)
  pickled_classifier.close()
  return classifier

def test_ner(input_file, classifier):
  pos_tagger = train_pos_tagger()
  input = codecs.open(input_file, 'rb','utf-8')
  for line in input:
    line = line[:-1]
    if len(line.strip()) == 0:
      continue
    for sent in sent_tokenize(line):
      inside_flag = "I,"
      beginning_flag = "B,"
      print_flag = beginning_flag
      tokens_sent = word_tokenize(re.sub(r"\"",'', re.sub(r"\'.",'', sent)).lower())
      pos_tagged = pos_tagger.tag(tokens_sent)
      io_tags = []
      for idx, (word, pos) in enumerate(pos_tagged):
        io_tags.append(classifier.classify(word_features(pos_tagged, idx)))
      ner_sent = zip(tokens_sent, io_tags)
      print_sent = []
      for token, io_tag in ner_sent:
        if io_tag == True:
          stop = stopwords.words('english') + list(string.punctuation)
          if token not in stop:
          		print_sent.append(" " + print_flag + " ")
          		print_flag = inside_flag
          else:
            print_sent.append(" " + "O," + " ")
            print_flag = beginning_flag
        else:
          print_sent.append(" " + "O," + " ")
          print_flag = beginning_flag
      print "[" + " ".join(print_sent) + "]"
      print_sent = []
      for token, io_tag in ner_sent:
        print_sent.append(" " + token + ", ")
      print "[" + " ".join(print_sent) + "]"
      print("\n")
      print("------------------------------------------------------------------------------")

  input.close()

def train_pos_tagger():
  """
  Trains a POS tagger with sentences from Penn Treebank
  and returns it.
  """
  train_sents = treebank.tagged_sents(tagset='universal')
  tagger = nltk.TrigramTagger(train_sents, backoff=
    nltk.BigramTagger(train_sents, backoff=
    nltk.UnigramTagger(train_sents, backoff=
    nltk.DefaultTagger("NN"))))
  return tagger

def ce_phrases():
  """
  Returns a list of phrases found using bootstrap.py ordered
  by number of words descending (so code traversing the list
  will encounter the longest phrases first).
  """
  def by_phrase_len(x, y):
    stop = stopwords.words('english') + list(string.punctuation)
    lx = len([i for i in word_tokenize(re.sub(r"\"",'', re.sub(r"\'.",'', x)).lower()) if i not in stop])
    ly = len([i for i in word_tokenize(re.sub(r"\"",'', re.sub(r"\'.",'', y)).lower()) if i not in stop])
    if lx == ly:
      return 0
    elif lx < ly:
      return 1
    else:
      return -1
  ceps = []
  phrasefile = codecs.open("sentence_features_train.txt", 'rb','utf-8')
  for cep in phrasefile:
    ceps.append(cep[:-1])
  phrasefile.close()
  stop = stopwords.words('english') + list(string.punctuation)
  return map(lambda phrase: ([i for i in word_tokenize(re.sub(r"\"",'', re.sub(r"\'.",'', phrase)).lower()) if i not in stop]),
    sorted(ceps, cmp=by_phrase_len))

def ce_phrases_test():
  """
  Returns a list of phrases found using bootstrap.py ordered
  by number of words descending (so code traversing the list
  will encounter the longest phrases first).
  """
  def by_phrase_len(x, y):
    stop = stopwords.words('english') + list(string.punctuation)
    lx = len([i for i in word_tokenize(re.sub(r"\"",'', re.sub(r"\'.",'', x)).lower()) if i not in stop])
    ly = len([i for i in word_tokenize(re.sub(r"\"",'', re.sub(r"\'.",'', y)).lower()) if i not in stop])
    if lx == ly:
      return 0
    elif lx < ly:
      return 1
    else:
      return -1
  ceps = []
  phrasefile = codecs.open("sentence_features_test.txt", 'rb','utf-8')
  for cep in phrasefile:
    ceps.append(cep[:-1])
  phrasefile.close()
  stop = stopwords.words('english') + list(string.punctuation)
  return map(lambda phrase: ([i for i in word_tokenize(re.sub(r"\"",'', re.sub(r"\'.",'', phrase)).lower()) if i not in stop]),
    sorted(ceps, cmp=by_phrase_len))
    

def ce_phrase_words(ce_phrases):
  """
  Returns a set of words in the ce_phrase list. This is
  used to tag words that refer to the NE but does not
  have a consistent pattern to match against.
  """
  ce_words = set()
  for ce_phrase_tokens in ce_phrases:
    for ce_word in ce_phrase_tokens:
      ce_words.add(ce_word)
  return ce_words

def slice_matches(a1, a2):
  """
  Returns True if the two arrays are content wise identical,
  False otherwise.
  """
  if len(a1) != len(a2):
    return False
  else:
    for i in range(0, len(a1)):
      if a1[i] != a2[i]:
        return False
    return True
  
def slots_available(matched_slots, start, end):
  """
  Returns True if all the slots in the matched_slots array slice
  [start:end] are False, ie, available, else returns False.
  """
  return len(filter(lambda slot: slot, matched_slots[start:end])) == 0

def promote_coreferences(tuple, ce_words):
  """
  Sets the io_tag to True if it is not set and if the word is
  in the set ce_words. Returns the updated tuple (word, pos, iotag)
  """
  return (tuple[0], tuple[1],
    True if tuple[2] == False and tuple[0] in ce_words else tuple[2])

def tag(sentence, pos_tagger, ce_phrases, ce_words):
  """
  Tokenizes the input sentence into words, computes the part of
  speech and the IO tag (for whether this word is "in" a CE named
  entity or not), and returns a list of (word, pos_tag, io_tag)
  tuples.
  """
  stop = stopwords.words('english') + list(string.punctuation)
  tokens = [i for i in word_tokenize(re.sub(r"\'.",'', sentence).lower()) if i not in stop]
  # add POS tags using our trained POS Tagger
  pos_tagged = pos_tagger.tag(tokens)
  # add the IO(not B) tags from the phrases we discovered
  # during bootstrap.
  words = [w for (w, p) in pos_tagged]
  pos_tags = [p for (w, p) in pos_tagged]
  io_tags = map(lambda word: False, words)
  for ce_phrase in ce_phrases:
    start = 0
    while start < len(words):
      end = start + len(ce_phrase)
      if slots_available(io_tags, start, end) and \
          slice_matches(words[start:end], ce_phrase):
        for j in range(start, end):
          io_tags[j] = True
        start = end + 1
      else:
        start = start + 1
  # zip the three lists together
  pos_io_tagged = map(lambda ((word, pos_tag), io_tag):
    (word, pos_tag, io_tag), zip(zip(words, pos_tags), io_tags))
  # "coreference" handling. If a single word is found which is
  # contained in the set of words created by our phrases, set
  # the IO(not B) tag to True if it is False
  return map(lambda tuple: promote_coreferences(tuple, ce_words),
    pos_io_tagged)



def word_features(tagged_sent, wordpos):
  return {
    "word": tagged_sent[wordpos][0],
    "pos": tagged_sent[wordpos][1],
    "prevword": "<START>" if wordpos == 0 else tagged_sent[wordpos-1][0],
    "prevpos": "<START>" if wordpos == 0 else tagged_sent[wordpos-1][1],
    "nextword": "<END>" if wordpos == len(tagged_sent)-1
                        else tagged_sent[wordpos+1][0],
    "nextpos": "<END>" if wordpos == len(tagged_sent)-1
                       else tagged_sent[wordpos+1][1],
    
  }

      
def main():
  if len(sys.argv) != 2:
    print "Usage ./cener.py [train|test]"
    sys.exit(-1)
  if sys.argv[1] == "train":
    classifier = train_ner("ce_ner_classifier.pkl")
  else:
  	print "NaiveBayesClassifier: Results\n\n"
  	classifier = get_trained_ner("NaiveBayesClassifierce_ner_classifier.pkl")
  	test_ner("sentence_test.txt", classifier)
  	print "DecisionTreeClassifier: Results\n\n"
  	classifier = get_trained_ner("DecisionTreeClassifierce_ner_classifier.pkl")
  	test_ner("sentence_test.txt", classifier)
  	print "MaxentClassifier: Results\n\n"
  	classifier = get_trained_ner("MaxentClassifierce_ner_classifier.pkl")
  	test_ner("sentence_test.txt", classifier)  
  	
if __name__ == "__main__":
  main()
