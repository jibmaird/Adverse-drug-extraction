#!/usr/bin/env python

import os, sys
import numpy as np
import optparse
import loader
import json
from nn import forward_with_bias

from utils import *
from loader import update_tag_scheme, prepare_dataset
from model import Model
import copy

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
    "-m", "--model", default="./Golden/test_model_True_False_52053",
    help="Model location"
)
opts = optparser.parse_args()[0]

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

bias = [0.0]*19

#model = Model(model_path=opts.model)
_, _, _, f_elements = model.build(training=False, **parameters)
model.reload()

dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
trn_sentences = loader.load_sentences(opts.train, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)
update_tag_scheme(trn_sentences, tag_scheme)

#print(parameters)

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
                             dev_data, id_to_tag, None, bias)
obs_tst, trans_tst = get_obs_trans(feat_dict, test_sentences, parameters, f_elements, test_sentences,
                              test_data, id_to_tag, None, bias)
obs_trn, trans_trn = get_obs_trans(feat_dict, trn_sentences, parameters, f_elements, trn_sentences,
                              trn_data, id_to_tag, None, bias)

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
                    id_to_tag, f_precal, obs_dev, trans_dev, bias)
print("Original dev scores:", dev_score_org)
tst_score_org = evaluate_constraints(parameters, test_sentences, test_data,
                    id_to_tag, f_precal, obs_tst, trans_tst, bias)
print("Original test scores:", tst_score_org)


avr = 5
best_trn = 0
best_dev = 0
new_test = 0
dev2_score_best = 0
best_candidate = None
candidates = [bias]
step = 0.05

num_of_bags = 1
index = range(len(dev_sentences))
index_bags = np.array_split(index, num_of_bags)
    
list_tag = [tag_to_id['O']]
print(list_tag)

list_best = []
list_tag=[]
tags = ['O']

for tag in tags:
    list_tag.append(tag_to_id[tag])
    
for bag_ID in range(len(index_bags)):
    best_dev = 0
    new_test = 0
    best_candidate = None
    candidates = [bias]
    patience = 4
    
    dev_bag_ID = index_bags[bag_ID]
    cur_dev_sentences = [dev_sentences[x] for x in dev_bag_ID]
    cur_dev_data = [dev_data[x] for x in dev_bag_ID]
    cur_obs_dev = [obs_dev[x] for x in dev_bag_ID]
    cur_trans_dev = [trans_dev[x] for x in dev_bag_ID]
    
    for tag in list_tag:
        print("Start tag : {}", tag)
        new_candidates = {}
        for candidate in candidates:
            best_tag_bias = 0.00
            tag_bias = 0.5
            while tag_bias < 1.51:
                sys.stdout.flush()
                new_candidate = copy.deepcopy(candidate)
                new_candidate[tag] = tag_bias
                print("===")
                print("decode with bias: " + ', '.join('%5.4f' % v for v in new_candidate))
                
                new_candidate[tag] = tag_bias
                trn_score = evaluate_constraints(parameters, trn_sentences, trn_data,
                            id_to_tag, f_precal, obs_trn, trans_trn, new_candidate)
                dev_score = evaluate_constraints(parameters, cur_dev_sentences, cur_dev_data,
                            id_to_tag, f_precal, cur_obs_dev, cur_trans_dev, new_candidate)
                test_score = 0
                if tag == list_tag[-1]:
                    test_score = evaluate_constraints(parameters, test_sentences, test_data,
                            id_to_tag, f_precal, obs_tst, trans_tst, new_candidate)
                new_candidate_str = json.dumps(new_candidate)
                new_candidates[new_candidate_str] = dev_score
                
                if dev_score > best_dev:
                    best_trn = trn_score
                    new_test = test_score
                    best_dev = dev_score
                    best_candidate = new_candidate
                    patience = 4
                else:
                    patience -= 1
                    if patience < 0:
                        break
                    
                tag_bias += step
                
        print("========================================================================================")
        print("Number of candidates: ", len(new_candidates.values()))
        sorted_new_candidates = sorted(new_candidates.items(), key=operator.itemgetter(1), reverse=True)
        candidates=[json.loads(x[0]) for x in sorted_new_candidates[:beam_size]]
        print(new_candidates.values())
        print([x[1] for x in sorted_new_candidates])
        print(candidates)
        print("========================================================================================")
        print("Search Depth: ", tag)
        print('Best score on dev: ', best_dev)
        print("Associated score on test: ", new_test)
        print("Best candidate: ", best_candidate)
        print("========================================================================================")
        
        list_best.append(best_candidate)