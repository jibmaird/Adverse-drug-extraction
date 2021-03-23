#!/usr/bin/perl -wd

use strict;

use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/AE_literature/orig_corpus";

#read relations
open(I,"/home/jibmaird/Data/Corpora/ADE-v2/DRUG-AE.rel")||die;
undef my %R;
while(<I>) {
    chomp;
    my @F = split(/\|/,$_);

    open(I2,"$data_d/$F[0]\.txt")||die;
    my $str = "";

    while(<I2>) {
	$str .= $_;
    }
    close(I2);
    print "$str\n";
    my $aux = substr $str,$F[3];
    print "$aux\n";
    last;
}
close(I);
