#!/usr/bin/perl -w

use strict;
use vars qw ();

chdir("/home/jibmaird/Projects/NLP_pipelines/stanford-corenlp-python");
system("python corenlp.py");
