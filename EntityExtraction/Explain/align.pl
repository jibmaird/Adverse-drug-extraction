#!/usr/bin/perl -w

use strict;

use vars qw ();

#read training
my $train_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Train-sent";
opendir(D,"$train_d")||die;
my @Train = grep /\.txt/,readdir D;
closedir(D);

#read test errors
my $error_d = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc/training-JskfW8vWg";

chdir("/home/jibmaird/Projects/Explainability/aligners/monolingual-word-aligner");

open(I1,"$error_d/errors.txt")||die;
open(I2,"$error_d/errors_id.txt")||die;

open(LOG,">$error_d/log.txt")||die;
while(<I1>) {
    chomp;
    my $err_sent = $_;
    my $tmp_f1 = "/home/jibmaird/Data/Corpora/ADE-v2/target_sent.txt";
    my $tmp_f2 = "/home/jibmaird/Data/Corpora/ADE-v2/target_sent2.txt";
    my $id = <I2>;
    chomp($id);

    foreach my $tr (@Train) {
	system("cp $train_d/$tr $tmp_f1");
	open(O,">$tmp_f2")||die;
	print O "$err_sent\n";
	close(O);
	$_ = `python testAlign.py`;
	chomp;
	print LOG "$id $tr $_\n";
    }
}
close(I1);
close(I2);
close(LOG);
