#!/usr/bin/perl -w

use strict;

use vars qw ();

#my $BIOBERT_DIR = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed";
my $BIOBERT_DIR = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc";
my $NER_DIR = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Biobert/bio";
my $RESULT_DIR = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Results";

#my $model_d = "training-Q4SZNHDWR";
#my $model_d = "training-BLUPtdDWR";
#my $model_d = "training-ydBtCfDWR";
my $model_d = "training-JskfW8vWg";
#my $model_f = "model.ckpt-2571";
my $model_f = "model.ckpt-1662";

#test
chdir("/home/jibmaird/Projects/biobert");
#system("python run_ner.py --do_train=false --do_predict=true --do_eval=true --vocab_file=$BIOBERT_DIR\/vocab.txt --bert_config_file=$BIOBERT_DIR\/bert_config.json --init_checkpoint=$BIOBERT_DIR/$model_d/$model_f --data_dir=$NER_DIR\/ --output_dir=$BIOBERT_DIR/$model_d");

#eval entity-level
system("python biocodes/ner_detokenize.py --token_test_path=$BIOBERT_DIR/$model_d/token_test.txt --label_test_path=$BIOBERT_DIR/$model_d/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$BIOBERT_DIR/$model_d");
#system("perl ./biocodes/conlleval.pl < $BIOBERT_DIR/$model_d/NER_result_conll.txt > $BIOBERT_DIR/$model_d/eval_entity.txt");
