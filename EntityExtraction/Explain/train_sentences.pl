#!/usr/bin/perl -w

use strict;

use vars qw ();

my $data_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper";

if (not -e "$data_d/Train-sent") {
    system("mkdir $data_d/Train-sent");
}
if (not -e "$data_d/Train-sent-ann") {
    system("mkdir $data_d/Train-sent-ann");
}

opendir(D,"$data_d/Train")||die;
my @File = grep /\.txt/,readdir D;
closedir(D);

chdir("/home/jibmaird/Projects/ADE/EntityExtraction/brat/tools");
foreach my $f (@File) {
    $_ = `python sentencesplit.py < $data_d/Train/$f`;
    my @F = split(/\n/,$_);
    my $i = 0;
    #read annotations
    $f =~ /^(.*?)\.txt/;
    open(I,"$data_d/Train/$1.ann")||die;
    undef my %Ann;
    while(<I>) {
	chomp;
	my @A = split(/\t/,$_);
	if ($A[1] =~/AdverseEvent (\d+) (\d+)/) {
	    
	    $Ann{"$1\#$2\#$A[2]"} = 1;
	}
    }
    close(I);

    my $start = 0;my $end = 0;
    foreach (@F) {
	$end = $start+length($_);
	next if $_ eq "";
	$f =~s /\.txt//;
	open(O,">$data_d/Train-sent/$f\.$i\.txt")||die;
	print O "$_\n";
	close(O);
	undef my %H;
	foreach my $ann (keys %Ann) {
	    (my $as,my $ae,$ann) = split(/\#/,$ann);

	    if ((/\b$ann\b/)&&($as>=$start)&&($ae<=$end)) {
		$H{$ann}=1;
	    }
	}
	$_ = join "\t",sort keys %H;
	open(O,">$data_d/Train-sent-ann/$f\.$i\.txt")||die;
	print O "$_\n";
	close(O);

	$i++;
	$start = $end;
    }
}
