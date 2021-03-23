import os, sys
import re
import numpy as np
import scipy.io
import theano
import theano.tensor as T
import codecs
import pickle
import numpy as np
from theano.gof.graph import inputs

from utils import shared, set_values, get_name, max_length, n_candidates
from nn import HiddenLayer, EmbeddingLayer, DropoutLayer, LSTM, forward, forward_with_bias
from optimization import Optimization

from theano.ifelse import ifelse

class Model(object):
    """
    Network architecture.
    """
    def __init__(self, parameters=None, models_path=None, model_path=None):
        """
        Initialize the model. We either provide the parameters and a path where
        we store the models, or the location of a trained model.
        """
        if model_path is None:
            assert parameters and models_path
            # Create a name based on the parameters
            self.parameters = parameters
            self.name = get_name(parameters)
            # Model location
            model_path = os.path.join(models_path, self.name)
            #model_path = "/".join([models_path,self.name])
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Create directory for the model if it does not exist
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            # Save the parameters to disk
            with open(self.parameters_path, 'wb') as f:
                pickle.dump(parameters, f)
        else:
            assert parameters is None and models_path is None
            # Model location
            self.model_path = model_path
            self.parameters_path = os.path.join(model_path, 'parameters.pkl')
            self.mappings_path = os.path.join(model_path, 'mappings.pkl')
            # Load the parameters and the mappings from disk
            with open(self.parameters_path, 'rb') as f:
                self.parameters = pickle.load(f)
            self.reload_mappings()
            self.load_bias()
        self.components = {}
        self.type="stack 3"

    def load_bias(self):
        self.bias = os.path.join(self.model_path, 'bias.txt')
        bias = []
        if os.path.isfile(self.bias):
            f = open(self.bias,'r')
            line = f.read().splitlines()[0]
            line = line.replace("[","")
            line = line.replace("]","")
            parts = line.split(", ")
            for part in parts:
                bias.append(float(part))
            self.bias = bias
        else:
            self.bias = [1.0]*(len(self.id_to_tag)+2)

    def set_manual_feature_map(self,feature_map,length):
        self.manual_map = feature_map
        self.manual_len = length

    def save_mappings(self, id_to_word, id_to_char, id_to_tag):
        """
        We need to save the mappings if we want to use the model later.
        """
        self.id_to_word = id_to_word
        self.id_to_char = id_to_char
        self.id_to_tag = id_to_tag
        with open(self.mappings_path, 'wb') as f:
            mappings = {
                'id_to_word': self.id_to_word,
                'id_to_char': self.id_to_char,
                'id_to_tag': self.id_to_tag,
            }
            pickle.dump(mappings, f)

    def reload_mappings(self):
        """
        Load mappings from disk.
        """
        with open(self.mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        self.id_to_word = mappings['id_to_word']
        self.id_to_char = mappings['id_to_char']
        self.id_to_tag = mappings['id_to_tag']

    def add_component(self, param):
        """
        Add a new parameter to the network.
        """
        if param.name in self.components:
            raise Exception('The network already has a parameter "%s"!'
                            % param.name)
        self.components[param.name] = param

    def save(self):
        """
        Write components values to disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            if hasattr(param, 'params'):
                param_values = {p.name: p.get_value() for p in param.params}
            else:
                param_values = {name: param.get_value()}
            scipy.io.savemat(param_path, param_values)

    def reload(self):
        """
        Load components values from disk.
        """
        for name, param in self.components.items():
            param_path = os.path.join(self.model_path, "%s.mat" % name)
            param_values = scipy.io.loadmat(param_path)
            if hasattr(param, 'params'):
                for p in param.params:
                    set_values(p.name, p, param_values[p.name])
            else:
                set_values(name, param, param_values[name])



    def build(self,
              dropout,
              char_dim,
              char_lstm_dim,
              char_bidirect,
              word_dim,
              word_lstm_dim,
              word_bidirect,
              lr_method,
              pre_emb,
              crf,
              manual,
              cap_dim,
              isLM,
              isRevLM,
              costConf,
              zoc_value,
              zoh_value,
              training=True,
              LM_input_dim=1024,
              **kwargs
              ):
        """
        Build the network.
        """
        # Training parameters
        n_words = len(self.id_to_word)
        n_chars = len(self.id_to_char)
        n_tags = len(self.id_to_tag)

        # Number of capitalization features
        if cap_dim:
            n_cap = 4

        # Network variables
        word_LM = T.fmatrix(name='word LM')
        word_LM_rev = T.fmatrix(name='word_LM_rev')
        is_train = T.iscalar('is_train')
        word_ids = T.ivector(name='word_ids')
        char_for_ids = T.imatrix(name='char_for_ids')
        char_rev_ids = T.imatrix(name='char_rev_ids')
        char_pos_ids = T.ivector(name='char_pos_ids')
        tag_ids = T.ivector(name='tag_ids')
        manual_features = T.fmatrix(name="manual_features")

        if cap_dim:
            cap_ids = T.ivector(name='cap_ids')

        # Sentence length
        s_len = (word_ids if word_dim else char_pos_ids).shape[0]

        # Final input (all word features)
        input_dim = 0
        inputs = []

        # Word inputs
        if word_dim:
            input_dim += word_dim
            word_layer = EmbeddingLayer(n_words, word_dim, name='word_layer')
            word_input = word_layer.link(word_ids)
            inputs.append(word_input)
            # Initialize with pretrained embeddings
            if pre_emb and training:
                new_weights = word_layer.embeddings.get_value()
                print('Loading pretrained embeddings from %s...' % pre_emb)
                pretrained = {}
                emb_invalid = 0
                for i, line in enumerate(codecs.open(pre_emb, 'r', 'utf-8', errors='ignore')):
                #for i, line in enumerate(codecs.open(pre_emb, 'r', 'latin-1', errors='ignore')):
                    line = line.rstrip().split()
                    if len(line) == word_dim + 1:
                        pretrained[line[0]] = np.array(
                            [float(x) for x in line[1:]]
                        ).astype(np.float32)
                    else:
                        emb_invalid += 1
                if emb_invalid > 0:
                    print('WARNING: %i invalid lines' % emb_invalid)
                c_found = 0
                c_lower = 0
                c_zeros = 0
                # Lookup table initialization
                for i in range(n_words):
                    word = self.id_to_word[i]
                    if word in pretrained:
                        new_weights[i] = pretrained[word]
                        c_found += 1
                    elif word.lower() in pretrained:
                        new_weights[i] = pretrained[word.lower()]
                        c_lower += 1
                    elif re.sub('\d', '0', word.lower()) in pretrained:
                        new_weights[i] = pretrained[
                            re.sub('\d', '0', word.lower())
                        ]
                        c_zeros += 1
                    # uncomment to print the missing words
                    ##else:
                    ##    print(word + "not found")
                word_layer.embeddings.set_value(new_weights)
                print('Loaded %i pretrained embeddings.' % len(pretrained))
                print(('%i / %i (%.4f%%) words have been initialized with '
                       'pretrained embeddings.') % (
                            c_found + c_lower + c_zeros, n_words,
                            100. * (c_found + c_lower + c_zeros) / n_words
                      ))
                print(('%i found directly, %i after lowercasing, '
                       '%i after lowercasing + zero.') % (
                          c_found, c_lower, c_zeros
                      ))

        #
        # Chars inputs
        #
        if char_dim:
            input_dim += char_lstm_dim
            char_layer = EmbeddingLayer(n_chars, char_dim, name='char_layer')

            char_lstm_for = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_for')
            char_lstm_rev = LSTM(char_dim, char_lstm_dim, with_batch=True,
                                 name='char_lstm_rev')

            char_lstm_for.link(char_layer.link(char_for_ids))
            char_lstm_rev.link(char_layer.link(char_rev_ids))

            char_for_output = char_lstm_for.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]
            char_rev_output = char_lstm_rev.h.dimshuffle((1, 0, 2))[
                T.arange(s_len), char_pos_ids
            ]

            inputs.append(char_for_output)
            if char_bidirect:
                inputs.append(char_rev_output)
                input_dim += char_lstm_dim

        #
        # Capitalization feature
        #
        if cap_dim:
            input_dim += cap_dim
            cap_layer = EmbeddingLayer(n_cap, cap_dim, name='cap_layer')
            inputs.append(cap_layer.link(cap_ids))

        if isLM:
            inputs.append(word_LM)
            input_dim += LM_input_dim
        
        if isRevLM:
            inputs.append(word_LM_rev)
            input_dim += 512

        # Prepare final input
        if len(inputs) != 1:
            inputs = T.concatenate(inputs, axis=1)

        #
        # Dropout on final input
        #

        if dropout:
            dropout_layer = DropoutLayer(p=dropout)
            input_train = dropout_layer.link(inputs)
            input_test = (1 - dropout) * inputs
            inputs = T.switch(T.neq(is_train, 0), input_train, input_test)

        #if isLM:
            #input_with_LM = T.concatenate([inputs,word_LM],axis=1)
        #else:
            #input_with_LM = inputs

        # LSTM for words
        word_lstm_for = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for', is_train=is_train, zoc_value=zoc_value, zoh_value=zoh_value)
        word_lstm_rev = LSTM(input_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev', is_train=is_train, zoc_value=zoc_value, zoh_value=zoh_value)
        word_lstm_for.link(inputs)
        word_lstm_rev.link(inputs[::-1, :])

        word_for_output = word_lstm_for.h
        word_rev_output = word_lstm_rev.h[::-1, :]

        if word_bidirect:
            final_output = T.concatenate(
                [word_for_output, word_rev_output],
                axis=1
            )
            tanh_layer = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer', activation=None)
            final_output = tanh_layer.link(final_output)
        else:
            final_output = word_for_output

        #Residual layer
        #########
        #########
        #########
        #########
        #BEGIN OF RESIDUAL

        stack_input = T.concatenate([inputs,final_output],axis=1)

        word_lstm_for_res = LSTM(input_dim+word_lstm_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_for_res', is_train=is_train, zoc_value=zoc_value, zoh_value=zoh_value)
        word_lstm_rev_res = LSTM(input_dim+word_lstm_dim, word_lstm_dim, with_batch=False,
                             name='word_lstm_rev_res', is_train=is_train, zoc_value=zoc_value, zoh_value=zoh_value)
        word_lstm_for_res.link(stack_input)
        word_lstm_rev_res.link(stack_input[::-1, :])
        word_for_output_res = word_lstm_for_res.h
        word_rev_output_res = word_lstm_rev_res.h[::-1, :]
        if word_bidirect:
            final_output_res = T.concatenate(
                [word_for_output_res, word_rev_output_res],
                axis=1
            )
            tanh_layer_res = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                     name='tanh_layer_res', activation='relu')
            final_output_res = tanh_layer_res.link(final_output_res)
        else:
            final_output_res = word_for_output_res

        #END OF RESIDUAL
        #########
        #########
        #########
        #########
        #########

        # Another Residual layer
        #########
        #########
        #########
        #########
        # BEGIN OF RESIDUAL

        stack_input_2 = T.concatenate([inputs, final_output_res], axis=1)

        word_lstm_for_res_2 = LSTM(input_dim + word_lstm_dim, word_lstm_dim, with_batch=False,
                                 name='word_lstm_for_res_2', is_train=is_train, zoc_value=zoc_value, zoh_value=zoh_value)
        word_lstm_rev_res_2 = LSTM(input_dim + word_lstm_dim, word_lstm_dim, with_batch=False,
                                 name='word_lstm_rev_res_2', is_train=is_train, zoc_value=zoc_value, zoh_value=zoh_value)
        word_lstm_for_res_2.link(stack_input_2)
        word_lstm_rev_res_2.link(stack_input_2[::-1, :])
        word_for_output_res_2 = word_lstm_for_res_2.h
        word_rev_output_res_2 = word_lstm_rev_res_2.h[::-1, :]
        if word_bidirect:
            final_output_res_2 = T.concatenate(
                [word_for_output_res_2, word_rev_output_res_2],
                axis=1
            )
            tanh_layer_res_2 = HiddenLayer(2 * word_lstm_dim, word_lstm_dim,
                                         name='tanh_layer_res_2', activation='relu')
            final_output_res_2 = tanh_layer_res_2.link(final_output_res_2)
        else:
            final_output_res_2 = word_for_output_res_2

        stack_input_3 = T.concatenate([inputs, final_output_res_2], axis=1)
        final_size = input_dim + word_lstm_dim

        #if isLM:
            #stack_input_3 = T.concatenate([stack_input_3,word_LM],axis=1)
            #final_size += 1024

        # Sentence to Named Entity tags - Score

        #
        # Dropout on stack_input_3
        #
        #if dropout:
        #    dropout_layer = DropoutLayer(p=0.8)
        #    input_train = dropout_layer.link(stack_input_3)
        #    input_test = (1 - dropout) * stack_input_3
        #    stack_input_3 = T.switch(T.neq(is_train, 0), input_train, input_test)

        final_layer_res_2 = HiddenLayer(final_size, n_tags, name='final_layer_2',
                                      # activation=(None if crf else 'softmax')
                                      activation=(None))
        tags_scores = final_layer_res_2.link(stack_input_3)


        # END OF RESIDUAL 2
        #########
        #########
        #########
        #########
        #########

        if manual:
            man_feat_dimreduce_layer = HiddenLayer(self.manual_len, word_lstm_dim, name='manual_feature_dim_reduction',
                                      activation='tanh')
            man_feat_reduced = man_feat_dimreduce_layer.link(manual_features)

            #final_output = T.concatenate([final_output,man_feat_reduced],axis=1)

            final_layer_manual = HiddenLayer(word_lstm_dim, n_tags, name='final_layer_manual',
                                      activation=(None))

            tags_scores_manual = final_layer_manual.link(man_feat_reduced)
            tags_scores = tags_scores + tags_scores_manual

        # No CRF
        if not crf:
            cost = T.nnet.categorical_crossentropy(tags_scores, tag_ids).mean()

        #CRF
        else:
            transitions = shared((n_tags + 2, n_tags + 2), 'transitions')

            small = -1000
            b_s = np.array([[small] * n_tags + [0, small]]).astype(np.float32)
            e_s = np.array([[small] * n_tags + [small, 0]]).astype(np.float32)
            observations = T.concatenate(
                [tags_scores, small * T.ones((s_len, 2))],
                axis=1
            )
            observations = T.concatenate(
                [b_s, observations, e_s],
                axis=0
            )
            # Score from tags
            real_path_score = tags_scores[T.arange(s_len), tag_ids].sum()

            # Score from transitions
            b_id = theano.shared(value=np.array([n_tags], dtype=np.int32))
            e_id = theano.shared(value=np.array([n_tags + 1], dtype=np.int32))

            padded_tags_ids = T.concatenate([b_id, tag_ids, e_id], axis=0)
            real_path_score += transitions[
                padded_tags_ids[T.arange(s_len + 1)],
                padded_tags_ids[T.arange(s_len + 1) + 1]
            ].sum()

            all_paths_scores = forward(observations, transitions)
            ## Testing confidence penalty
            if not costConf>0:
              cost = - (real_path_score - all_paths_scores)
            else:
              p_y_s = real_path_score - all_paths_scores
              #p_yn_s = 1.0000000001 - T.exp(p_y_s)
              #p_yn_s = ifelse(T.lt(p_yn_s, 0.0000000001), 0.0000000001, p_yn_s)
              cost = - (p_y_s + (costConf * (T.exp(p_y_s) * p_y_s))) # + p_yn_s * T.log(p_yn_s))))

        # Network parameters
        # NORMAL PARAMS
        params = []
        params_reduced = []
        if word_dim:
            self.add_component(word_layer)
            params.extend(word_layer.params)
        if char_dim:
            self.add_component(char_layer)
            self.add_component(char_lstm_for)
            params.extend(char_layer.params)
            params.extend(char_lstm_for.params)
            if char_bidirect:
                self.add_component(char_lstm_rev)
                params.extend(char_lstm_rev.params)
        self.add_component(word_lstm_for)
        params.extend(word_lstm_for.params)

        self.add_component(word_lstm_for_res)
        params.extend(word_lstm_for_res.params)

        self.add_component(word_lstm_for_res_2)
        params.extend(word_lstm_for_res_2.params)

        if word_bidirect:
            self.add_component(word_lstm_rev)
            params.extend(word_lstm_rev.params)

            self.add_component(word_lstm_rev_res)
            params.extend(word_lstm_rev_res.params)

            self.add_component(word_lstm_rev_res_2)
            params.extend(word_lstm_rev_res_2.params)

        if cap_dim:
            self.add_component(cap_layer)
            params.extend(cap_layer.params)

        # If there are manual features, add layers, params
        if manual:
            self.add_component(man_feat_dimreduce_layer)
            params.extend(man_feat_dimreduce_layer.params)
            self.add_component(final_layer_manual)
            params.extend(final_layer_manual.params)
        # End of manual

        params_reduced=params[:]
        self.add_component(final_layer_res_2)
        params.extend(final_layer_res_2.params)

        if crf:
            self.add_component(transitions)
            params.append(transitions)
        else:
            self.add_component(transitions)
            params.append(transitions)

        if word_bidirect:

            self.add_component(tanh_layer)
            self.add_component(tanh_layer_res)
            self.add_component(tanh_layer_res_2)

            params.extend(tanh_layer.params)
            params.extend(tanh_layer_res.params)
            params.extend(tanh_layer_res_2.params)

        # MANUAL PARAMS

        # Prepare train and eval inputs
        eval_inputs = []

        if isLM:
            eval_inputs.append(word_LM)
        if isRevLM:
            eval_inputs.append(word_LM_rev)

        if manual:
            eval_inputs.append(manual_features)
        if word_dim:
            eval_inputs.append(word_ids)
        if char_dim:
            eval_inputs.append(char_for_ids)
            if char_bidirect:
                eval_inputs.append(char_rev_ids)
            eval_inputs.append(char_pos_ids)
        if cap_dim:
            eval_inputs.append(cap_ids)
        if crf:
            train_inputs = eval_inputs + [tag_ids]
        else:
            train_inputs = eval_inputs + [tag_ids]
            #train_inputs = eval_inputs
            #train_inputs = train_inputs + twisted_ids + F1_scores
        bias=T.vector("bias")
        eval_inputs.append(bias)

        # Parse optimization method parameters
        if "-" in lr_method:
            lr_method_name = lr_method[:lr_method.find('-')]
            lr_method_parameters = {}
            for x in lr_method[lr_method.find('-') + 1:].split('-'):
                split = x.split('_')
                assert len(split) == 2
                lr_method_parameters[split[0]] = float(split[1])
        else:
            lr_method_name = lr_method
            lr_method_parameters = {}

        # Compile training function

        sys.stdout.flush()
        #print('Compiling...')
        sys.stdout.flush()
        f_eval_bias = None
        f_train = None
        if training:

            updates = Optimization(clip=5.0).get_updates(lr_method_name, cost, params, **lr_method_parameters)

            f_train = theano.function(
                inputs=train_inputs,
                outputs=cost,
                updates=updates,
                givens=({is_train: np.cast['int32'](1)} if dropout else {})
            )

        # Compile evaluation function
        if not crf:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=tags_scores,
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
        else:
            f_eval = theano.function(
                inputs=eval_inputs,
                outputs=forward(observations, transitions, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {}),
                on_unused_input='ignore'
            )
            f_eval_bias = theano.function(
                inputs=eval_inputs,
                outputs=forward_with_bias(observations, transitions, bias=bias, viterbi=True,
                                return_alpha=False, return_best_sequence=True),
                givens=({is_train: np.cast['int32'](0)} if dropout else {})
            )
            
            f_elements = theano.function(
                inputs=eval_inputs,
                outputs=[observations, transitions],
                givens=({is_train: np.cast['int32'](0)} if dropout else {}),
                on_unused_input='ignore'
            )

        return f_train, f_eval, f_eval_bias, f_elements
