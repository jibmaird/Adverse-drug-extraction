#!/usr/bin/perl -wd

use strict;

use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/AE_literature/orig_corpus";

#read orig corpus
undef my %H;
opendir(D,$data_d)||die;
my @File = grep /\.ann/,readdir D;
closedir(D);

foreach my $f (@File) {
    open(I,"$data_d/$f")||die;
    $f =~s /\....$//;
    while(<I>) {
	chomp;
	/^T(\d+)\s+.*?(\d+) (\d+)\t(.*)$/;
	$H{$f}{"$2\-$3\-$4"} = $1;
    }
    close(I);
}

#read relations
open(I,"/home/jibmaird/Data/Corpora/ADE-v2/DRUG-AE.rel")||die;
undef my %R;
while(<I>) {
    chomp;
    my @F = split(/\|/,$_);
    if (not defined $H{$F[0]}) {
	print "NOT FOUND: $F[0]\n";
    }
    else {
	if (not defined $H{$F[0]}{"$F[3]\-$F[4]\-$F[2]"}) {
	    print "NOT FOUND: $F[0] $F[3]\-$F[4]\-$F[2]\n";
	}
    }
}
close(I);
