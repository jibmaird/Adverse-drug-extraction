#!/usr/bin/perl -w

use strict;
use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/AE_literature/manual_annotation/annotated";
my $out_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper";

opendir(D,"$data_d")||die;
my @File = grep /\.csv/,readdir D;
closedir(D);

undef my %H;
foreach my $f (@File) {
    open(I,"$data_d/$f")||die;
    while(<I>) {
	chomp;
	my @F = split(/\,/,$_);
	if ((defined $F[4])&&($F[4] =~s /.*\///)) {
	    $H{$F[4]} = 1;
	}
    }
    close(I);
}

my $data_d2 = "/home/jibmaird/Data/Corpora/AE_literature/orig_corpus";
opendir(D,"$data_d2")||die;
@File = grep /\.txt/,readdir D;
closedir(D);

my $checked = 0;
my $total = 0;
open(O,">$out_d/checked.txt")||die;
open(O2,">$out_d/all.txt")||die;
foreach my $f (@File) {
    $f=~s /\.txt//;
    if (defined $H{$f}) {
	$checked++;
	print O "$f\n";
    }
    print O2 "$f\n";
    $total++;
}
print "CHECKED: $checked\n";
print "TOTAL: $total\n";
close(O);
close(O2);
