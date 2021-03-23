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

train_d = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg/Sentence-embed/"

Files = os.listdir(train_d)
regex = re.compile(r'.*\.txt')
Files = list(filter(regex.search,Files))

for f in Files:
    with open(train_d+f) as json_f:
        data = json.load(json_f)
        v1 = data['features'][0]['layers'][0]['values']
        a = np.array(v1)
        error_d = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg/Errors-embed/"
        Files2 = os.listdir(error_d)
        regex = re.compile(r'.*\.txt')
        Files2 = list(filter(regex.search,Files2))
        for f2 in Files2:
            with open(error_d+f2) as json_f2:
                data2 = json.load(json_f2)
                v2 = data2['features'][0]['layers'][0]['values']
                #print(f+" "+f2)
                #print(data['features'][0]['token'])
                #print(data2['features'][0]['token'])
                b = np.array(v2)
                # manually compute cosine similarity
                dot = np.dot(a, b)
                norma = np.linalg.norm(a)
                normb = np.linalg.norm(b)
                cos = dot / (norma * normb)
                print(cos)
