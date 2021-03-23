#!/usr/bin/perl -w

use strict;

use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Biobert";

system("cat $data_d/conll/train.tsv $data_d/conll/train_dev.tsv > $data_d/conll3files/train.tsv");
system("cp $data_d/conll/test.tsv $data_d/conll3files/test.tsv");
system("cp $data_d/conll/test_dev.tsv $data_d/conll3files/devel.tsv");
