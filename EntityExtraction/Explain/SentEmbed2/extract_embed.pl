#!/usr/bin/perl -w

use strict;

use vars qw ();

#For each sentence in training and test, extract the last layer from the trained model
my $train_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Train-sent";
opendir(D,"$train_d")||die;
my @Train = grep /\.txt/,readdir D;
closedir(D);

chdir("/home/jibmaird/Projects/biobert");
my $BERT_BASE_DIR = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg";
my $out_d = "$BERT_BASE_DIR/Sentence-embed";
if (not -e $out_d) {
    system("mkdir $out_d");
}

#foreach my $f (@Train) {
#    print "$f\n";
#    my $in_f = "$train_d/$f";
#    my $out_f = "$out_d/$f";
#    system("python extract_features.py --input_file=$in_f --output_file=$out_f --vocab_file=$BERT_BASE_DIR/../vocab.txt --bert_config_file=$BERT_BASE_DIR/../bert_config.json --init_checkpoint=$BERT_BASE_DIR/model.ckpt-1662 --layers=-1 --max_seq_length=128 --batch_size=8");
#}

my $errors_f = "$BERT_BASE_DIR/errors.txt";

if (not -e "$BERT_BASE_DIR/Errors"){
    system("mkdir $BERT_BASE_DIR/Errors");
}
if (not -e "$BERT_BASE_DIR/Errors-embed"){
    system("mkdir $BERT_BASE_DIR/Errors-embed");
}

open(I,"$errors_f")||die;
open(I2,"$BERT_BASE_DIR/errors_id.txt")||die;


while(<I>) {
    my $i = <I2>;
    chomp($i);
    chomp;
    open(O,">$BERT_BASE_DIR/Errors/$i.txt")||die;
    print O "$_\n";
    close(O);
    system("python extract_features.py --input_file=$BERT_BASE_DIR/Errors/$i.txt --output_file=$BERT_BASE_DIR/Errors-embed/$i.txt --vocab_file=$BERT_BASE_DIR/../vocab.txt --bert_config_file=$BERT_BASE_DIR/../bert_config.json --init_checkpoint=$BERT_BASE_DIR/model.ckpt-1662 --layers=-1 --max_seq_length=128 --batch_size=8");
}
close(I);
close(I2);
