import os,sys
import re
import codecs
import numpy as np
import operator
import theano
import theano.tensor as T
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
import random
import copy, pickle
from collections import Counter
import os.path

from sklearn.metrics import f1_score
from itertools import zip_longest

models_path = "./models"
eval_path = "./evaluation"
eval_temp = os.path.join(eval_path, "temp")
eval_script = os.path.join(eval_path, "conlleval")
max_length = 200
n_tags = 17
n_candidates = 5
threshold = 0.9

def get_name(parameters):
    """
    Generate a model name from its parameters.
    """
    l = []
    for k, v in parameters.items():
        if type(v) is str and "/" in v:
            l.append((k, v[::-1][:v[::-1].index('/')][::-1]))
        else:
            l.append((k, v))
    name = ",".join(["%s=%s" % (k, str(v).replace(',', '')) for k, v in l])
    name = "test_model_"+str(parameters['crf'])+"_"+str(parameters["manual"])+"_"+str(int(random.random()*100000))
    return "".join(i for i in name if i not in "\/:*?<>|")


def set_values(name, param, pretrained):
    """
    Initialize a network parameter with pretrained values.
    We check that sizes are compatible.
    """
    param_value = param.get_value()
    if pretrained.size != param_value.size:
        raise Exception(
            "Size mismatch for parameter %s. Expected %i, found %i."
            % (name, param_value.size, pretrained.size)
        )
    param.set_value(np.reshape(
        pretrained, param_value.shape
    ).astype(np.float32))


def shared(shape, name):
    """
    Create a shared object of a numpy array.
    """
    if len(shape) == 1:
        value = np.zeros(shape)  # bias are initialized with zeros
    else:
        drange = np.sqrt(6. / (np.sum(shape)))
        value = drange * np.random.uniform(low=-1.0, high=1.0, size=shape)
    return theano.shared(value=value.astype(theano.config.floatX), name=name)


def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    Check that tags have a valid IOB format.
    Tags in IOB1 format are converted to IOB2.
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    """
    Replace singletons by the unknown word with a probability p.
    """
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def pad_word_chars(words):
    """
    Pad the characters of the words in a sentence.
    Input:
        - list of lists of ints (list of words, a word being a list of char indexes)
    Output:
        - padded list of lists of ints
        - padded list of lists of ints (where chars are reversed)
        - list of ints corresponding to the index of the last character of each word
    """
    max_length = max([len(word) for word in words])
    char_for = []
    char_rev = []
    char_pos = []
    for word in words:
        padding = [0] * (max_length - len(word))
        char_for.append(word + padding)
        char_rev.append(word[::-1] + padding)
        char_pos.append(len(word) - 1)
    return char_for, char_rev, char_pos


def create_input(feature_dict, train_sentence, data, parameters, add_label, singletons=None, LMsource=None, LMRevsource=None):
    """
    Take sentence data and return an input for
    the training or the evaluation function.
    """
    words = data['words']
    chars = data['chars']

    if singletons is not None:
        words = insert_singletons(words, singletons)
    if parameters['cap_dim']:
        caps = data['caps']
    char_for, char_rev, char_pos = pad_word_chars(chars)
    input = []


    if parameters["isLM"]:
        input.append(LMsource)
    if parameters["isRevLM"]:
        input.append(LMRevsource)
    if parameters['manual']:
        man = feature_dict.transform(sent2features(train_sentence))
        input.append(man)
    if parameters['word_dim']:
        input.append(words)
    if parameters['char_dim']:
        input.append(char_for)
        if parameters['char_bidirect']:
            input.append(char_rev)
        input.append(char_pos)
    if parameters['cap_dim']:
        input.append(caps)
    if add_label:
        input.append(data['tags'])
    return input

def get_obs_trans(feat_dict, sentences, parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag, dictionary_tags, bias=[1.0]*19, wordLM=None, wordLMRev=None):
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)
    index = 0
    obs = []
    trans = []
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        if parameters['isLM']:
            if parameters['isRevLM']:
                input = create_input(feat_dict, sentences[index], data, parameters, False,
                    LMsource=wordLM[index],LMRevsource=wordLMRev[index])
            else:
                input = create_input(feat_dict, sentences[index], data, parameters, False, LMsource=wordLM[index], LMRevsource=None)
        else:
            input = create_input(feat_dict, sentences[index], data, parameters, False, LMsource=None, LMRevsource=None)
        input.append(bias)
        index += 1
        #if index % int(len(raw_sentences)/10) == 0:
            #print("Precompute %d sentences in %d sentences" % (index, len(raw_sentences)))
        #    sys.stdout.flush()
        data = f_eval(*input)
        obs.append(np.asarray(data[0]))
        trans.append(np.asarray(data[1]))
    return obs, trans

def evaluate(feat_dict, sentences, parameters, f_eval, raw_sentences, parsed_sentences,
             id_to_tag,forwardLM=None,backwardLM=None):

    f_error = open('error.txt','w')
    """
    Evaluate current model using CoNLL script.
    """
    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)
    index = 0
    #assert len(raw_sentences) == len(wordLM)
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        LM_rev_PLH = LM_PLH = None
        if parameters['isLM']:
            LM_PLH = forwardLM[index]
        if parameters['isRevLM']:
            LM_rev_PLH = backwardLM[index]
        input = create_input(feat_dict, sentences[index], data, parameters, False,
                LMsource=LM_PLH,LMRevsource=LM_rev_PLH)
        
        index += 1
        if parameters['crf']:
            y_preds = np.array(f_eval(*input,bias=[1.0]*n_tags))[1:-1]
        else:
            y_preds = f_eval(*input,bias=[1.0]*n_tags).argmax(axis=1)
        y_reals = np.array(data['tags']).astype(np.int32)

        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(1000000, 2000000)
    output_path = os.path.join(eval_temp, "eval.%i.output" % eval_id)
    scores_path = os.path.join(eval_temp, "eval.%i.scores" % eval_id)
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    f.close()

    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    for line in eval_lines:
        print(line)

    # Remove temp files
    # os.remove(output_path)
    # os.remove(scores_path)

    # Confusion matrix with accuracy for each tag

    '''
    print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in xrange(n_tags)] + ["Percent"])
    )
    for i in xrange(n_tags):
        print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in xrange(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        )'''

    # Global accuracy
    print("%i/%i (%.5f%%)" % (
        count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
    ))
    f_error.close()
    # F1 on all entities
    return float(eval_lines[1].strip().split()[-1])

def output_constraints(parameters, raw_sentences, parsed_sentences,
             id_to_tag, f_precal, f_precal2, f_precal3, observations, transitions, bias, isPrint=True, ID=0):

    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)
    index = 0
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        y_preds = np.array(f_precal(observations[index], transitions[index], bias)[1:-1])
        #y_preds2 = np.array(f_precal2(observations[index], transitions[index], bias)[1:-1])
        
        y_preds2 = np.array(f_precal2(observations[index], transitions[index], bias))
        y_preds3 = np.array(f_precal3(observations[index], transitions[index], bias))
        print(y_preds)
        print(y_preds2)
        print(y_preds3)
        index += 1
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    print("\n".join(predictions))

def evaluate_constraints(parameters, raw_sentences, parsed_sentences,
             id_to_tag, f_precal, observations, transitions, bias, isPrint=True, ID=0):

    n_tags = len(id_to_tag)
    predictions = []
    count = np.zeros((n_tags, n_tags), dtype=np.int32)
    index = 0
    for raw_sentence, data in zip(raw_sentences, parsed_sentences):
        y_preds = np.array(f_precal(observations[index], transitions[index], bias)[1:-1])
        index += 1
        y_reals = np.array(data['tags']).astype(np.int32)
        assert len(y_preds) == len(y_reals)
        p_tags = [id_to_tag[y_pred] for y_pred in y_preds]
        r_tags = [id_to_tag[y_real] for y_real in y_reals]
        if parameters['tag_scheme'] == 'iobes':
            p_tags = iobes_iob(p_tags)
            r_tags = iobes_iob(r_tags)
        for i, (y_pred, y_real) in enumerate(zip(y_preds, y_reals)):
            new_line = " ".join(raw_sentence[i][:-1] + [r_tags[i], p_tags[i]])
            predictions.append(new_line)
            count[y_real, y_pred] += 1
        predictions.append("")

    # Write predictions to disk and run CoNLL script externally
    eval_id = np.random.randint(10000000000, 20000000000)
    output_path = os.path.join(eval_temp, "eval.%i.%i.output" % (eval_id, ID))
    scores_path = os.path.join(eval_temp, "eval.%i.%i.scores" % (eval_id, ID))
    while os.path.isfile(output_path) or os.path.isfile(scores_path):
        eval_id = np.random.randint(10000000000, 20000000000)
        output_path = os.path.join(eval_temp, "eval.%i.%i.output" % (eval_id, ID))
        scores_path = os.path.join(eval_temp, "eval.%i.%i.scores" % (eval_id, ID))
        
    with codecs.open(output_path, 'w', 'utf8') as f:
        f.write("\n".join(predictions))
    os.system("%s < %s > %s" % (eval_script, output_path, scores_path))

    # CoNLL evaluation results
    eval_lines = [l.rstrip() for l in codecs.open(scores_path, 'r', 'utf8')]
    if isPrint:
        for line in eval_lines:
            print(line)

    # Remove temp files
    os.remove(output_path)
    os.remove(scores_path)

    # Confusion matrix with accuracy for each tag

    '''
    print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in xrange(n_tags)] + ["Percent"])
    )
    for i in xrange(n_tags):
        print ("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * n_tags)).format(
            str(i), id_to_tag[i], str(count[i].sum()),
            *([count[i][j] for j in xrange(n_tags)] +
              ["%.3f" % (count[i][i] * 100. / max(1, count[i].sum()))])
        )'''

    # Global accuracy
    if isPrint:
        print("%i/%i (%.5f%%)" % (
            count.trace(), count.sum(), 100. * count.trace() / max(1, count.sum())
        ))
    # F1 on all entities
    return float(eval_lines[1].strip().split()[-1])

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    #print("sent: ",sent)
    #print("word: ",word)

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def corpusfeaturesdict(sents):
    word_feats = []
    for sent in sents:
        word_feats.extend(sent2features(sent))
    v = DictVectorizer(sparse=False)
    v.fit(word_feats)
    return v

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def get_embs_LM_pkl(source):
    filelist = [fn for fn in os.listdir(source) if os.path.isfile(os.path.join(source,fn))]
    filelist.sort()
    print(source, len(filelist))
    sentences = []
    for filename in filelist:
        data = pickle.load(file=open(os.path.join(source,filename),"rb"))
        data = np.reshape(data, newshape=(data.shape[0], data.shape[-1]))
        data = data[1:]
        sentences.append(data)
    return sentences
