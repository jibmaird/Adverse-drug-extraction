#!/usr/bin/perl -w

use strict;

use vars qw ();

open(I,"ADE-train-orig.yaml")||die;
open(O,">ADE-train.yaml")||die;
while(<I>) {
    chomp;
    s /biobert_v1.0_pubmed/biobert_v1.0_pubmed_pmc/g;
    print O "$_\n";
}
close(I);
close(O);
