#!/usr/bin/perl -w

use strict;

use vars qw ();

my $in_f = "";

chdir("/home/jibmaird/Projects/ADE/EntityExtraction/brat/tools");
system("python sentencesplit.py < /home/jibmaird/Data/Corpora/ADE-v2/Exper/Train/18707772.txt");
