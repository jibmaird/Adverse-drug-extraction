#!/usr/bin/perl -w

use strict;

use vars qw ();
my $biobert_d = "/home/jibmaird/Projects/biobert";

#training
system("cp ADE-train.yaml $biobert_d");
chdir($biobert_d);
system("zip biobert.zip *.py ADE-train.yaml");
system("bx ml train biobert.zip ADE-train.yaml");

#testing
#system("cp ADE-test.yaml $biobert_d");
#chdir($biobert_d);
#system("zip biobert.zip *.py ADE-test.yaml");
#system("bx ml train biobert.zip ADE-test.yaml");
