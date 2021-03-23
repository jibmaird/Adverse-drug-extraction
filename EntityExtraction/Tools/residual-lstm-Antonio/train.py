import os, sys
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader

from utils import *
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained
from model import Model
#from model_stack_2 import Model
#from model_stack_3 import Model
#from model_stack_4 import Model

import logging

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger("train")

# Read parameters from command line
optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="1",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Dropout on the input (0 = no dropout)"
)
optparser.add_option(
    "-L", "--lr_method", default="sgd-lr_.005",
    help="Learning method (SGD, Adadelta, Adam..)"
)
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-m", "--manual", default="0",
    type='int', help="Whether or not use the manual features"
)
optparser.add_option(
    "-X", "--batch", default="0",
    type='int', help="whether use batch"
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
    "-G", "--withLM",default="0",
    type='int',help="with LM or not"
)

optparser.add_option(
    "-K", "--withRevLM",default="0",
    type='int',help='with reverse LM or not'
)

optparser.add_option(
    "-P", "--models_path",default="./models",
    help='directory where to store models'
)

optparser.add_option(
    "-e", "--epochs",default="100",
    type='int', help='number of epochs'
)

optparser.add_option(
    "-F", "--costConf", default="0.0",
    type='float', help="confidence penalty"
)

optparser.add_option(
    "-x", "--zoc_value", default="1.0",
    type='float', help="zoc value"
)

optparser.add_option(
    "-o", "--zoh_value", default="1.0",
    type='float', help="zoh value"
)

opts = optparser.parse_args()[0]

# Set up models folder
models_path = opts.models_path

# Parse parameters
parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['lr_method'] = opts.lr_method
parameters['manual'] = opts.manual == 1
parameters['batch'] = opts.batch == 1
parameters['isLM'] = opts.withLM == 1
parameters['isRevLM'] = opts.withRevLM == 1

parameters['models_path'] = opts.models_path
parameters['epochs'] = opts.epochs == 1
parameters['costConf'] = opts.costConf

parameters['zoh_value'] = opts.zoh_value
parameters['zoc_value'] = opts.zoc_value

print(parameters)

# Check parameters validity
assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

LM_input_dim = 0

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
	
# Check evaluation script / folders
if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

# Initialize model
model = Model(parameters=parameters, models_path=models_path)
print("Model location: %s" % model.model_path)
print("Model type: %s" % model.type)

# Data parameters
lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

# Load sentences
train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)

# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

# Create a dictionary / mapping of words
# If we use pretrained embeddings, we add them to the dictionary.
if parameters['pre_emb']:
    dico_words_train = word_mapping(train_sentences, lower)[0]
    dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
else:
    dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
    dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
feat_dict = corpusfeaturesdict(train_sentences)
#print feat_dict.transform(word2features(train_sentences[0],0))
manual_length = len(feat_dict.transform(word2features(train_sentences[0],0))[0])

print(manual_length)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

singletons = set([word_to_id[k] for k, v in dico_words_train.items() if v == 1])

# Save the mappings to disk
print('Saving the mappings to disk...')
model.save_mappings(id_to_word, id_to_char, id_to_tag)
model.set_manual_feature_map(feature_map=feat_dict, length=manual_length)

# Build the model
print (LM_input_dim)
f_train, f_eval, f_eval_bias, f_elements = model.build(**parameters,LM_input_dim=LM_input_dim)

# Reload previous model values
if opts.reload:
    print('Reloading previous model...')
    model.reload()

sys.stdout.flush()
#
# Train network
#
#n_epochs = 300  # number of epochs over the training set
n_epochs = opts.epochs  # number of epochs over the training set
freq_eval = len(train_data) #int(len(train_data)/20)
#freq_eval = int(len(train_data)/1000)
freq_print = int(len(train_data)/7)
mem_length = freq_print
best_dev = -np.inf
best_test = -np.inf
real_best_tes = -np.inf
count = 0
train_mode = 0
patience = 0
max_patience = 2
for epoch in range(n_epochs):
    if epoch == 0:
        epoch_costs = [0.0]*mem_length
    epoch_costs = epoch_costs[-mem_length:]
    print("Starting epoch %i..." % epoch)
    for i, index in enumerate(np.random.permutation(len(train_data))):
        count += 1
        input = create_input(feat_dict, train_sentences[index], train_data[index], parameters, True, singletons)
        new_cost = f_train(*input)
        epoch_costs.append(new_cost)

        if i % freq_print == 0 and i > 0 == 0:
            print("%i\%i, cost average (mem: %i): %f, best score test: %f" % (i, len(train_data), mem_length, np.mean(epoch_costs[-mem_length:]), real_best_tes))
            sys.stdout.flush()
        if epoch > 10:
            freq_eval = int(len(train_data)/20)
        if count % freq_eval == 0:
            ## Added try to avoid exceptions: ajimeno - 20170921
            try:
              dev_score = evaluate(feat_dict, dev_sentences, parameters, f_eval, dev_sentences,
                                   dev_data, id_to_tag, dico_tags)
              #test_score = dev_score
              test_score = evaluate(feat_dict, test_sentences, parameters, f_eval, test_sentences,
                                    test_data, id_to_tag, dico_tags)
              print("Score on dev: %.5f" % dev_score)
              print("Score on test: %.5f" % test_score)
              if dev_score > best_dev:
                  best_dev = dev_score
                  print("New best score on dev.")
                  print("Saving model to disk...")
                  real_best_tes = test_score
                  model.save()
              if test_score > best_test:
                  best_test = test_score
                  print("New best score on test.")
            except Exception as e:
              print ("Error while processing the evaluation process")
              logger.error(e, exc_info=True)

    print("Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs)))

