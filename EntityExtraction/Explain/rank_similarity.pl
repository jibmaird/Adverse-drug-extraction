#!/usr/bin/perl -w

use strict;

use vars qw ();

my $sentences = "/home/jibmaird/Data/Corpora/ADE-v2/all_sentences.txt";
my $log = "/home/jibmaird/Projects/Explainability/aligners/monolingual-word-aligner/log.txt";
my $ann = "/home/jibmaird/Data/Corpora/ADE-v2/DRUG-AE.rel";


undef my %H;
undef my @S;
undef my %A;

#DOC-LEVEL AE ANNOTATIONS

#REL ANNOTATIONS
open(I,"$ann")||die;
while(<I>) {
    chomp;
    my @F = split(/\|/,$_);
    $A{$F[1]}{$F[2]} = 1;
}
close(I);

open(I,"$sentences")||die;
my $i = 0;
while(<I>) {
    chomp;
    $S[$i] = $_;
    $i++;
}
close(I);

open(I,"$log")||die;
my $al = "";
while(<I>) {
    chomp;
    if (/^\[/) {
	$al = $_;
    }
    elsif (/line no (\d+) (\d+) computed with a score of (.*)$/) {
	$H{"$1\-$2"}{al} = $al;
	$H{"$1\-$2"}{s} = $3;
    }
}
close(I);

foreach my $ind (sort {$H{$b}{s}<=>$H{$a}{s}} keys %H) {
    print "$ind\t$H{$ind}{s}\t$H{$ind}{al}\n";
    $ind =~ /^(\d+)\-(\d+)$/;
    my $an1 = join "#",keys %{$A{$S[$1]}};
    my $an2 = join "#",keys %{$A{$S[$2]}};
    print "\t$S[$1]\#$an1\n";
    print "\t$S[$2]\#$an2\n";
}
