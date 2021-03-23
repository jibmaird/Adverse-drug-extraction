#!/usr/bin/perl -w

use strict;

use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper";
my $orig_d = "/home/jibmaird/Data/Corpora/AE_literature/orig_corpus";
my $manual_d = "/home/jibmaird/Data/Corpora/AE_literature/manual_annotation/annotated_ready/ann_04032019b";

open(I,"$data_d/random_test.txt")||die;
undef my %H;
while(<I>) {
    chomp;
    my @F = split(/\, \'/,$_);
    foreach (@F) {
	/(\d+)/;
	$H{$1} = 1;
    }
}
close(I);

open(I,"$data_d/all.txt")||die;
while(<I>) {
    chomp;
    #test
    if (defined $H{$_}) {
	system("cp $manual_d/$_\.txt $data_d/Test");
	system("cp $manual_d/$_\.ann $data_d/Test");
    }
    #train
    else {
	if (-e "$manual_d/$_") {
	    system("cp $manual_d/$_\.txt $data_d/Train");
	    system("cp $manual_d/$_\.ann $data_d/Train");
	}
	else {
	    system("cp $orig_d/$_\.txt $data_d/Train");
	    system("cp $orig_d/$_\.ann $data_d/Train");
	}
    }
}
close(I);
