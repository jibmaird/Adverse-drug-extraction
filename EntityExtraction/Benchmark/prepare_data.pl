#!/usr/bin/perl -w

use strict;
use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Biobert";

opendir(D,"$data_d/bio")||die;
my @File = grep /\.tsv$/,readdir D;
closedir(D);

foreach my $f (@File) {
    my $sent = 1;
    open(I,"$data_d/bio/$f")||die;
    open(O,">$data_d/benchmark/$f")||die;
    while(<I>) {
	chomp;
	if ($_ ne "") {
	    print O "$sent\t$_\n";
	}
	else {
	    $sent++;
	}
    }
    close(I);
    close(O);
}
