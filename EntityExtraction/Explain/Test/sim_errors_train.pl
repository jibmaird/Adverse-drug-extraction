#!/usr/bin/perl -w

use strict;

use vars qw ();




chdir("/home/jibmaird/Projects/Explainability/aligners/monolingual-word-aligner");
system("python testAlign.py");
