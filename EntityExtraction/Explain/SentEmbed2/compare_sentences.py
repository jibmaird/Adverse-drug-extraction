#!/usr/bin/python
# -*- coding: utf-8 -*-
# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# (C) Copyright IBM Corp. 2001, 2019. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

'''
Created on Oct 31, 2019

@author: DAVIDMARTINEZIRAOLA
'''

import json
import os
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys

#train_d = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg/Sentence-embed/"
train_d = sys.argv[1]
error_d = sys.argv[2]

Files = os.listdir(train_d)
regex = re.compile(r'.*\.txt')
Files = list(filter(regex.search,Files))

for f in Files:
    with open(train_d+f) as json_f:
        data = json.load(json_f)
        v1 = []
        for w in data['features']:
            if (not w['token']=="[CLS]"):
                #print(w['token'])
                v1.append(w['layers'][0]['values'])
        v1 = np.asarray(v1)
        v1 = np.average(v1,axis=0)

#        error_d = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg/Errors-embed/"
        Files2 = os.listdir(error_d)
        regex = re.compile(r'.*\.txt')
        Files2 = list(filter(regex.search,Files2))
        for f2 in Files2:
            with open(error_d+f2) as json_f2:
                data2 = json.load(json_f2)
                v2 = []
                for w in data2['features']:
                    if (not w['token']=="[CLS]"):
                        v2.append(w['layers'][0]['values'])
                v2 = np.asarray(v2)
                v2 = np.average(v2,axis=0)

                # manually compute cosine similarity
                dot = np.dot(v1, v2)
                norma = np.linalg.norm(v1)
                normb = np.linalg.norm(v2)
                cos = dot / (norma * normb)
                print(f+" "+f2+" "+str(cos))
