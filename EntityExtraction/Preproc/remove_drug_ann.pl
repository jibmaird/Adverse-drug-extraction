#!/usr/bin/perl -w

use strict;

use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Biobert/conll";

opendir(D,$data_d)||die;
my @File = grep /\.tsv$/,readdir D;
closedir(D);

foreach my $f (@File) {
    open(I,"$data_d/$f")||die;
    open(O,">$data_d/$f~")||die;
    while(<I>) {
	chomp;
	s /[BI]\-Drug\t/O\t/;
	print O "$_\n";
    }
    close(I);
    close(O);
    system("mv $data_d/$f~ $data_d/$f");
}
