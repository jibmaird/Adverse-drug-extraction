import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
#from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups

import os
import re

in_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper"

#read test

test_data = []
test_target = []

fname = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg/errors.txt"

with open(fname) as f:
            for line in f:
                test_data.append(line)

fname = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg/errors_tokens.txt"

with open(fname) as f:
            for line in f:
                p = re.compile('^.*\#B.*$')
                m = p.match(line)
                if m:
                    test_target.append(0)
                else:
                    test_target.append(1)

#read training

Files = os.listdir(in_d+"/Train-sent")

train_data = []
train_target = []
for fname in Files:
    with open(in_d+"/Train-sent/"+fname) as f:
            for line in f:
                train_data.append(line)
    with open(in_d+"/Train-sent-ann/"+fname) as f:
            for line in f:
                line = line.rstrip()
#                print line+"#"
                if line == "": 
                    train_target.append(0)
#                    print "0"
                else:
                    train_target.append(1)
#                    print "1"




vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, train_target)

pred = rf.predict(test_vectors)
sc = sklearn.metrics.f1_score(test_target, pred, average='binary')
print sc

from lime import lime_text
from sklearn.pipeline import make_pipeline
c = make_pipeline(vectorizer, rf)
print(c.predict_proba([test_data[0]]))

from lime.lime_text import LimeTextExplainer
class_names = ['0', '1']
explainer = LimeTextExplainer(class_names=class_names)

idx = 0
for item in test_data:
    exp = explainer.explain_instance(test_data[idx], c.predict_proba, num_features=6)
    print('Document id: %d' % idx)
    print('Sentence: %s' %test_data[idx])
    print('Probability(0) =', c.predict_proba([test_data[idx]])[0,0])
    print('True class: %s' % class_names[test_target[idx]])

    print(exp.as_list())
    print("\n\n");

#    print('Original prediction:', rf.predict_proba(test_vectors[idx])[0,0])

    idx = idx + 1

#tmp = test_vectors[idx].copy()
#tmp[0,vectorizer.vocabulary_['Posting']] = 0
#tmp[0,vectorizer.vocabulary_['Host']] = 0
#print('Prediction removing some features:', rf.predict_proba(tmp)[0,0])
#print('Difference:', rf.predict_proba(tmp)[0,0] - rf.predict_proba(test_vectors[idx])[0,0])
