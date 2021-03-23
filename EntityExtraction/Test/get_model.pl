#!/usr/bin/perl -w

#$m = $ARGV[0];
#$m = "training--966BmDWg";
#$m = "training-52_u5LvWg";
#$m = "training-1vqlOyvWg";
$m = "training-JskfW8vWg";

#my $BIOBERT_DIR = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed";
my $BIOBERT_DIR = "/home/jibmaird/Projects/biobert-pretrained/biobert_v1.0_pubmed_pmc";
system("mkdir $BIOBERT_DIR/$m");

system("aws --endpoint-url=http://s3.ap-geo.objectstorage.softlayer.net s3 cp s3://dmi-bucket/$m/ $BIOBERT_DIR/$m --recursive");
