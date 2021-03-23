#!/usr/bin/perl -w

use strict;

use vars qw ();


my $log_d = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg";
my $train_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Train-sent";
my $ann_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Train-sent-ann";
#my $log_f = "comp_log.txt";
my $log_f = $ARGV[0];

#read training annotations
undef my %Ann;
opendir(D,"$train_d")||die;
my @File = grep /\.txt/,readdir D;
closedir(D);
foreach my $f (@File) {
    open(I,"$train_d/$f")||die;
    my $sent = "";
    while(<I>) {
	chomp;
	$sent = $_;
    }
    close(I);
    open(I,"$ann_d/$f")||die;
    while(<I>) {
	chomp;
	$Ann{$sent} = $_;
    }
    close(I);
}

#read errors
undef my %E;
open(I,"$log_d/errors.txt")||die;
open(I2,"$log_d/errors_id.txt")||die;
open(I3,"$log_d/errors_tokens.txt")||die;
while(<I>) {
    chomp;
    my $id = <I2>;
    chomp($id);
    my $aux = <I3>;
    chomp($aux);
    $E{$id} = "$_\nPrediction: $aux";
}
close(I);
close(I2);
close(I3);

open(I,"$log_f")||die;
undef my %H;
my $comp = "";
my $exp = "";
while(<I>) {
    chomp;

    my @F = split(/ /,$_);
    $F[1] =~s /\.txt//;
    $H{$F[1]}{$F[0]}{sc} = $F[2];
}
close(I);

foreach my $s1 (keys %H) {
    my $i = 0;
    print "Error in $s1\:$E{$s1}\nComments:\n\n";
    foreach my $s2 (sort {$H{$s1}{$b}{sc}<=>$H{$s1}{$a}{sc}} keys %{$H{$s1}}) {
	open(I,"$train_d/$s2")||die;
	my $sent = "";
	while(<I>) {
	    chomp;
	    $sent = $_;
	}
	close(I);
	print "\tSimilar sentence $i: $s2 Score: $H{$s1}{$s2}{sc}\n\tSentence: $sent\n\tGold labels: $Ann{$sent}\n\tManual labels:\n\tComments:\n\n";
	$i++;
	last if $i == 10;
    }
}
