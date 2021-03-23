#!/usr/bin/perl -w

use strict;
use vars qw ();

use List::Util qw/shuffle/;

my $data_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper";

my $out_d = "$data_d/Biobert/conll";

if (not -e $out_d) {
    system("mkdir $out_d");}

my @T = ("train","test");

foreach my $t (@T) {

    my $T = ucfirst($t);

    opendir(D,"$data_d/$T")||die;
    my @File = grep /\.conll/,readdir D;
    closedir(D);
    
    @File = shuffle @File;

    my $out_f = "$out_d/$t\_dev.tsv";
    open(O,">$out_f")||die;
    my $i = 0;
    foreach my $f (@File) {
	open(I,"$data_d/$T/$f")||die;
	$i++;
	if ($i > $#File / 2) {
	    $i = -100;
	    close(O);
	    open(O,">$out_d/$t.tsv")||die;
	}
	while(<I>) {
	    print O "$_";
	}
	close(I);
	print O "\n";
    }
    close(O);
}
system("mv $out_d/test_dev.tsv $out_d/devel.tsv");
