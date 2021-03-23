#!/usr/bin/perl -w

use strict;

use vars qw ();

my $trans_d = "/home/jibmaird/Projects/ADE/EntityExtraction/Tools/sentence-transformers/examples";

#copy errors
my $errors_f = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg/errors.txt";
my $queries_tmp = "$trans_d/queries.txt";
system("cp $errors_f $queries_tmp");

#copy training
my $train_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Train-sent";
opendir(D,"$train_d")||die;
my @Train = grep /\.txt/,readdir D;
closedir(D);
open(O,">$trans_d/corpus.txt")||die;
foreach my $tr (@Train) {
    open(I,"$train_d/$tr")||die;
    while(<I>) {
	print O;
    }
    close(I);
}
close(O);

chdir($trans_d);
system("python application_semantic_search2.py"); #requires python3.7
