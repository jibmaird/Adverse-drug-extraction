#!/usr/bin/env python

import os
import time
import codecs
import optparse
import numpy as np
import sys
from loader import prepare_sentence
from utils import create_input, iobes_iob, zero_digits, get_embs_LM_pkl
from model import Model

optparser = optparse.OptionParser()

optparser.add_option(
    "-i", "--input", default="./text_files/test.txt",
    help="Input file location"
)
#optparser.add_option(
#    "-o", "--output", default="./text_files/output_test.txt",
#    help="Output file location"
#)
optparser.add_option(
    "-d", "--delimiter", default="__",
    help="Delimiter to separate words from their tags"
)
optparser.add_option(
    "-Y", "--LM_sens",default="./LM/eng_for/test/",
    help="train LM docs"
)
optparser.add_option(
    "-Q", "--LM_rev_sens",default="./LM/eng_rev/test/",
    help="train LM docs"
)

optparser.add_option(
    "-m", "--model", default="./Golden/WithLM/English_LM_Dual_9166/",
    help="Model location"
)

optparser.add_option(
    "-B", "--bias", default="",
    help="bias location"
)

opts = optparser.parse_args()[0]

# Check parameters validity
assert opts.delimiter
assert os.path.isdir(opts.model)
assert os.path.isfile(opts.input)

# Load existing model
sys.stderr.write("Loading model...\n")
model = Model(model_path=opts.model)
parameters = model.parameters

LM_input_dim = 0

if len(opts.bias) > 0:
    model.load_bias(opts.bias)

if ('isLM' not in parameters.keys()):
    parameters['isLM'] = False
if ('isRevLM' not in parameters.keys()):
    parameters['isRevLM'] = False

if parameters['isLM']:
    LM_sens = get_embs_LM_pkl(opts.LM_sens)
    LM_input_dim = LM_sens[0][0].shape[-1]
else:
    LM_rev_sens = None

if parameters['isRevLM']:
    LM_rev_sens = get_embs_LM_pkl(opts.LM_rev_sens)
else:
    LM_rev_sens = None

# Load reverse mappings
word_to_id, char_to_id, tag_to_id = [
    {v: k for k, v in x.items()}
    for x in [model.id_to_word, model.id_to_char, model.id_to_tag]
]

# Load the model
_, f_eval, f_eval_bias, _ = model.build(training=False, **parameters, LM_input_dim=LM_input_dim)
model.reload()

#f_output = codecs.open(opts.output, 'w', 'utf-8')
f_output = sys.stdout
start = time.time()

sys.stderr.write('Tagging...\n')
with codecs.open(opts.input, 'r', 'utf-8') as f_input:
    count = 0
    for line in f_input:
        words = line.rstrip().split()
        if line.rstrip():
            # Lowercase sentence
            if parameters['lower']:
                line = line.lower()
            # Replace all digits with zeros
            if parameters['zeros']:
                line = zero_digits(line)
            # Prepare input
            sentence = prepare_sentence(words, word_to_id, char_to_id,
                                        lower=parameters['lower'])

            #input = create_input(None, words, sentence, parameters, add_label=False, singletons=None, LMsource=LM_sens[count], LMRevsource=LM_rev_sens[count])
            input = create_input(None, words, sentence, parameters, add_label=False, singletons=None)

            #input = create_input(sentence, parameters, False)
            input.append(model.bias)

            # Decoding
            if parameters['crf']:
                y_preds = np.array(f_eval_bias(*input))[1:-1]
            else:
                y_preds = f_eval_bias(*input).argmax(axis=1)
            y_preds = [model.id_to_tag[y_pred] for y_pred in y_preds]

            # Output tags in the IOB2 format
            if parameters['tag_scheme'] == 'iobes':
                y_preds = iobes_iob(y_preds)
            # Write tags
            assert len(y_preds) == len(words)
            #f_output.write('%s\n' % ' '.join('%s%s%s' % (w, opts.delimiter, y)
            #                                for w, y in zip(words, y_preds)))

            f_output.write('%s\n' % ' '.join('%s' % (y)
                                             for w, y in zip(words, y_preds)))
            #f_output.write(''.join('%s %s\n' % (w, y)
            #               for w, y in zip(words, y_preds)))
        else:
            f_output.write('\n')
        count += 1
        if count % 100 == 0:
            sys.stderr.write(str(count))
            sys.stderr.write('\n')

sys.stderr.write('---- %i lines tagged in %.4fs ----' % (count, time.time() - start))
sys.stderr.write('\n')
f_output.close()
