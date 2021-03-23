#!/usr/bin/env python

import os, sys
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader
import json
import operator
from nn import forward_with_bias
import math

from utils import *
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset, prepare_batch_dataset
from loader import augment_with_pretrained
from model import Model
#from model_stack_2 import Model
#from model_withLM import Model
import copy
import random
from random import shuffle

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="./dataset/eng.train",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="./dataset/eng.testa",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="./dataset/eng.testb",
    help="Test set location"
)

optparser.add_option(
    "-Y", "--trainLM",default="./LM/eng_for/train/",
    help="train LM docs"
)
optparser.add_option(
    "-Z", "--devLM",default="./LM/eng_for/dev/",
    help="dev LM docs"
)
optparser.add_option(
    "-U", "--testLM",default="./LM/eng_for/test/",
    help="test LM docs"
)

optparser.add_option(
    "-Q", "--trainRevLM",default="./LM/eng_rev/train/",
    help="train LM docs"
)
optparser.add_option(
    "-V", "--devRevLM",default="./LM/eng_rev/dev/",
    help="dev LM docs"
)
optparser.add_option(
    "-E", "--testRevLM",default="./LM/eng_rev/test/",
    help="test LM docs"
)

optparser.add_option(
    "-m", "--model", default="./Golden/WithLM/test_model_True_False_50608/",
    help="Base Model location"
)

optparser.add_option(
    "-i", "--ID", default="0",
    type='int', help="process ID"
)

optparser.add_option(
    "-o", "--bias", default="",
    help="bias location"
)

opts = optparser.parse_args()[0]

eval_id = int(opts.ID)
# Check parameters validity
assert os.path.isdir(opts.model)

# Load existing model
print("Loading model...")
model = Model(model_path=opts.model)
parameters = model.parameters
#parameters['dropout'] = 0

feat_dict = None
word_dict = None

zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']
lower = parameters['lower']

LM_input_dim = 0

if('isLM' not in parameters.keys()):
    parameters['isLM'] = False
if('isRevLM' not in parameters.keys()):
    parameters['isRevLM'] = False
	
if parameters['isLM']:
    trainLM_sens = get_embs_LM_pkl(opts.trainLM)
    testLM_sens = get_embs_LM_pkl(opts.testLM)
    devLM_sens = get_embs_LM_pkl(opts.devLM)
    LM_input_dim = trainLM_sens[0][0].shape[-1]
else:
    trainLM_sens=testLM_sens=devLM_sens=None

if parameters['isRevLM']:
    trainRevLM_sens = get_embs_LM_pkl(opts.trainRevLM)
    testRevLM_sens = get_embs_LM_pkl(opts.testRevLM)
    devRevLM_sens = get_embs_LM_pkl(opts.devRevLM)
else:
    trainRevLM_sens=testRevLM_sens=devRevLM_sens=None

bias = [1.0]*19
_, _, _, f_elements = model.build(training=False, **parameters, LM_input_dim=LM_input_dim)
model.reload()

dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
trn_sentences = loader.load_sentences(opts.train, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)
update_tag_scheme(trn_sentences, tag_scheme)

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load sentences
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)
trn_data = prepare_dataset(
    trn_sentences, word_to_id, char_to_id, tag_to_id, lower
)

id_to_tag = model.id_to_tag
print(id_to_tag)

obs_dev, trans_dev = get_obs_trans(feat_dict, dev_sentences, parameters, f_elements, dev_sentences,
                             dev_data, id_to_tag, None, bias, devLM_sens, devRevLM_sens)
obs_tst, trans_tst = get_obs_trans(feat_dict, test_sentences, parameters, f_elements, test_sentences,
                              test_data, id_to_tag, None, bias, testLM_sens, testRevLM_sens)
obs_trn, trans_trn = get_obs_trans(feat_dict, trn_sentences, parameters, f_elements, trn_sentences,
                              trn_data, id_to_tag, None, bias, trainLM_sens, trainRevLM_sens)

OBS = T.matrix("obs")
TRANS = T.matrix("trans")
BIAS = T.vector("bias")
print("COMPILE VITERBI")
f_precal = theano.function(
    inputs=[OBS, TRANS, BIAS],
    outputs=forward_with_bias(observations=OBS,transitions=TRANS,bias=BIAS,viterbi=True,
                                return_alpha=False, return_best_sequence=True)
)
print("END COMPILE VITERBI")
sys.stdout.flush()

beam_size=1

dev_score_org = evaluate_constraints(parameters, dev_sentences, dev_data,
                    id_to_tag, f_precal, obs_dev, trans_dev, bias, ID=eval_id)
print("Original dev scores:", dev_score_org)
tst_score_org = evaluate_constraints(parameters, test_sentences, test_data,
                    id_to_tag, f_precal, obs_tst, trans_tst, bias, ID=eval_id)
print("Original test scores:", tst_score_org)


avr = 5
best_trn = 0
best_dev = 0
new_test = 0
dev2_score_best = 0
best_candidate = None
candidates = [bias]
step = 0.05

original_bias = copy.deepcopy(bias)


num_of_bags = 50
index = range(len(trn_sentences))
index_bags = list(np.array_split(index, num_of_bags))
    
list_tag = [tag_to_id['O']]
print(list_tag)

list_best = []
list_tag=[]
tags = ['O','S-ORG','S-LOC','B-PER','E-PER','I-ORG','B-ORG','E-ORG','I-MISC','S-PER','B-MISC','E-MISC','B-LOC','E-LOC','I-PER','S-MISC','I-LOC']

for tag in tags:
    list_tag.append(tag_to_id[tag])
test_score = 0
best_bias = bias

lr = 0.005
momentum = 0.9

import operator
from random import shuffle

bias_file = opts.bias
if len(bias_file) == 0:
    ran = random.randint(1000,9999)
    f_w = open(opts.model+"/bias."+str(ran)+".txt","w")
else:
    f_w = open(bias_file,"w")

for i in range(10):
    patience = 3
    index = list(range(len(trn_sentences)))
    shuffle(index)
    index_bags = list(np.array_split(index, num_of_bags))
    
    for batch in index_bags:
        grads = [0.0]*19
        prev_grad = [0.0]*19
        cur_trn_sentences = [trn_sentences[index] for index in batch]
        cur_trn_data = [trn_data[index] for index in batch]
        cur_obs_trn = [obs_trn[index] for index in batch]
        cur_trans_trn = [trans_trn[index] for index in batch]
        
        trn_score = evaluate_constraints(parameters, cur_trn_sentences, cur_trn_data,
                id_to_tag, f_precal, cur_obs_trn, cur_trans_trn, bias, isPrint=False, ID=eval_id)
        prev_bias = copy.deepcopy(bias)
        for tag in list_tag:
            list_tag_bias = [(m+0.0)/100 for m in range(1,11,1) if m != 0 ]
            shuffle(list_tag_bias)
            grad = 0
            list_tag_bias_pos = [list_tag_bias[0]]
            for tag_bias in list_tag_bias_pos:
                new_candidate = copy.deepcopy(bias)
                new_candidate_2 = copy.deepcopy(bias)
                new_candidate[tag] = new_candidate[tag] + tag_bias
                new_trn_score = evaluate_constraints(parameters, cur_trn_sentences, cur_trn_data,
                    id_to_tag, f_precal, cur_obs_trn, cur_trans_trn, new_candidate, isPrint=False, ID=eval_id)
                new_candidate_2[tag] = new_candidate_2[tag] - tag_bias
                new_trn_score_2 = evaluate_constraints(parameters, cur_trn_sentences, cur_trn_data,
                    id_to_tag, f_precal, cur_obs_trn, cur_trans_trn, new_candidate_2, isPrint=False, ID=eval_id)
                
                grad = (np.log2(1.0-new_trn_score/100)-np.log2(1.0-new_trn_score_2/100))/(2*tag_bias)
                if math.isnan(grad):
                    grad = 0.0
                grads[tag] += grad

        for tag in list_tag:
            grads[tag] = max(min(grads[tag], 2), -2)
            bias[tag] -= (lr * grads[tag]+momentum*prev_grad[tag])
            
        prev_grad = copy.deepcopy(grads)
            
        dev_score = evaluate_constraints(parameters, dev_sentences, dev_data,
            id_to_tag, f_precal, obs_dev, trans_dev, bias, isPrint=False, ID=eval_id)
        tst_score = evaluate_constraints(parameters, test_sentences, test_data,
            id_to_tag, f_precal, obs_tst, trans_tst, bias, isPrint=False, ID=eval_id)
        print("Iter %i with dev score: %5.5f\n" % (i, dev_score))
        sys.stdout.flush()
        
        if dev_score <  10:
            print("this bias is not stable: ", bias) 
            print("unstability - reset to previous value of bias")
            bias = copy.deepcopy(prev_bias)
        
        if dev_score > best_dev:
            best_bias = copy.deepcopy(bias)
            best_dev = dev_score
            test_score = evaluate_constraints(parameters, test_sentences, test_data,
                        id_to_tag, f_precal, obs_tst, trans_tst, bias, isPrint=False, ID=eval_id)
            print("=============================================")
            print("New best bias: ", best_bias)
            f_w.write(str(best_bias))
            print("Train score: ", trn_score)
            print("New best dev: ", best_dev)
            print("Associated test: ", test_score)
            print("=============================================")
            patience = 10
        else:
            if patience < 0 and i > 0:
                print("Final best bias: ", best_bias) 
                sys.exit(1)