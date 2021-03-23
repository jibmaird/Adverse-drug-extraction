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
from nn import forward_with_bias, forward
import math

from utils import *
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset, prepare_batch_dataset
from loader import augment_with_pretrained

from model_orig import Model

# from model_stack_2 import Model
# from model_withLM import Model
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
    "-Y", "--trainLM", default="./LM/eng_for/train/",
    help="train LM docs"
)
optparser.add_option(
    "-Z", "--devLM", default="./LM/eng_for/dev/",
    help="dev LM docs"
)
optparser.add_option(
    "-U", "--testLM", default="./LM/eng_for/test/",
    help="test LM docs"
)

optparser.add_option(
    "-Q", "--trainRevLM", default="./LM/eng_rev/train/",
    help="train LM docs"
)
optparser.add_option(
    "-V", "--devRevLM", default="./LM/eng_rev/dev/",
    help="dev LM docs"
)
optparser.add_option(
    "-E", "--testRevLM", default="./LM/eng_rev/test/",
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
    "-b", "--bias", default="",
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
if len(opts.bias) > 0:
    model.load_bias(opts.bias)
# parameters['dropout'] = 0

feat_dict = None
word_dict = None

zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']
lower = parameters['lower']

LM_input_dim = 0

if ('isLM' not in parameters.keys()):
    parameters['isLM'] = False
if ('isRevLM' not in parameters.keys()):
    parameters['isRevLM'] = False

if parameters['isLM']:
    trainLM_sens = get_embs_LM_pkl(opts.trainLM)
    testLM_sens = get_embs_LM_pkl(opts.testLM)
    devLM_sens = get_embs_LM_pkl(opts.devLM)
    LM_input_dim = trainLM_sens[0][0].shape[-1]
else:
    trainLM_sens = testLM_sens = devLM_sens = None

if parameters['isRevLM']:
    trainRevLM_sens = get_embs_LM_pkl(opts.trainRevLM)
    testRevLM_sens = get_embs_LM_pkl(opts.testRevLM)
    devRevLM_sens = get_embs_LM_pkl(opts.devRevLM)
else:
    trainRevLM_sens = testRevLM_sens = devRevLM_sens = None

bias = [1.0] * 19
_, f_eval, _, f_elements = model.build(training=False, **parameters, LM_input_dim=LM_input_dim)
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

bias = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

#obs_dev, trans_dev = get_obs_trans(feat_dict, dev_sentences, parameters, f_elements, dev_sentences,
#                                   dev_data, id_to_tag, None, bias, devLM_sens, devRevLM_sens)
obs_tst, trans_tst = get_obs_trans(feat_dict, test_sentences, parameters, f_elements, test_sentences,
                                   test_data, id_to_tag, None, bias, testLM_sens, testRevLM_sens)

OBS = T.matrix("obs")
TRANS = T.matrix("trans")
BIAS = T.vector("bias")
print("COMPILE VITERBI")
f_precal = theano.function(
    inputs=[OBS, TRANS, BIAS],
    outputs=forward_with_bias(observations=OBS, transitions=TRANS, bias=BIAS, viterbi=True,
                              return_alpha=False, return_best_sequence=True)
)
print("END COMPILE VITERBI")
sys.stdout.flush()

beam_size = 1
#bias = model.bias
#print("parameters")
#print(parameters)
#print("dev_sentences")
#print(dev_sentences)
#print("dev_data")
#print(dev_data)
#print("id_to_tag")
#print(id_to_tag)
#print("f_precal")
#print(f_precal)
#print("obs_dev")
#print(obs_dev)
#print("trans_dev")
#print(trans_dev)
#print("eval_id")
#print(eval_id)

#dev_score_org = evaluate_constraints(parameters, dev_sentences, dev_data,
#                                     id_to_tag, f_precal, obs_dev, trans_dev, bias, ID=eval_id)
#print("Original dev scores:", dev_score_org)
print(len(test_sentences))
print(len(test_data))
#print(f_precal.shape)
print(len(obs_tst))
print(len(trans_tst))

n = 20
batch=int(len(test_sentences)/n)

for c in range(0,n):
  begin=c*batch
  if c < n-1:
    end = begin + batch - 1
  else:
    end = len(test_sentences)-1  

  print(begin)
  print(end)
  tst_score_org = evaluate_constraints(parameters, test_sentences[begin:end], test_data[begin:end],
                                     id_to_tag, f_precal, obs_tst[begin:end], trans_tst[begin:end], bias, ID=eval_id)
  print("Batch original test scores:", tst_score_org)

tst_score_org = evaluate_constraints(parameters, test_sentences, test_data,
                                     id_to_tag, f_precal, obs_tst, trans_tst, bias, ID=eval_id)
print("Original test scores:", tst_score_org)
