#!/usr/bin/perl -w

use strict;

use vars qw ();

my $in_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Biobert/conll";
my $out_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Biobert/bio";

opendir(D,$in_d)||die;
my @File = grep /\w/,readdir D;
closedir(D);

foreach my $f (@File) {
    open(I,"$in_d/$f")||die;
    open(O,">$out_d/$f")||die;
    while(<I>) {
	chomp;
	my @F = split(/\t/,$_);
	if (/^([BI])\-AdverseEvent/) {
	    print O "$F[3]\t$1\n";
	}
	elsif ($_ ne "") {
	    print O "$F[3]\tO\n";
	}
	else {
	    print O "$_\n";
	}
    }
    close(I);
    close(O);
}
