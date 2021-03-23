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
# from model_stack_2 import Model
# from model_withLM import Model
import copy
import random
from random import shuffle

optparser = optparse.OptionParser()
optparser.add_option(
    "-t", "--input", default="./dataset/eng.testb",
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
#print("Loading model...")
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
    testLM_sens = get_embs_LM_pkl(opts.testLM)
    LM_input_dim = trainLM_sens[0][0].shape[-1]
else:
    testLM_sens = None

if parameters['isRevLM']:
    testRevLM_sens = get_embs_LM_pkl(opts.testRevLM)
else:
    testRevLM_sens = None

bias = [1.0] * 19
_, _, _, f_elements = model.build(training=False, **parameters, LM_input_dim=LM_input_dim)
model.reload()

test_sentences = loader.load_sentences(opts.input, lower, zeros)

update_tag_scheme(test_sentences, tag_scheme)

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
    ]

# Load sentences
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

id_to_tag = model.id_to_tag
#print(id_to_tag)

obs_tst, trans_tst = get_obs_trans(feat_dict, test_sentences, parameters, f_elements, test_sentences,
                                   test_data, id_to_tag, None, bias, testLM_sens, testRevLM_sens)

OBS = T.matrix("obs")
TRANS = T.matrix("trans")
BIAS = T.vector("bias")
#print("COMPILE VITERBI")
f_precal = theano.function(
    inputs=[OBS, TRANS, BIAS],
    outputs=forward_with_bias(observations=OBS, transitions=TRANS, bias=BIAS, viterbi=True,
                              return_alpha=False, return_best_sequence=True)
)
f_precal2 = theano.function(
    inputs=[OBS, TRANS, BIAS],
    outputs=forward_with_bias(observations=OBS, transitions=TRANS, bias=BIAS, viterbi=True,
                              return_alpha=False, return_best_sequence=False)
)
f_precal3 = theano.function(
    inputs=[OBS, TRANS, BIAS],
    outputs=forward_with_bias(observations=OBS, transitions=TRANS, bias=BIAS, viterbi=False,
                              return_alpha=False, return_best_sequence=False)
)

#get f_precal2 with return_best_sequence=False return_alpha=False

#print("END COMPILE VITERBI")
sys.stdout.flush()

beam_size = 1
bias = model.bias
output_constraints(parameters, test_sentences, test_data,
                                     id_to_tag, f_precal, f_precal2, f_precal3, obs_tst, trans_tst, bias, ID=eval_id)
