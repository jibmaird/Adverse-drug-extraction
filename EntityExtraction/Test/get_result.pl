#!/usr/bin/perl -w

#$m = $ARGV[0];
$m = "training-ydBtCfDWR";

my $DATA_DIR = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Results";
system("mkdir $DATA_DIR/$m");

system("aws --endpoint-url=http://s3.ap-geo.objectstorage.softlayer.net s3 cp s3://dmi-bucket/$m/ $DATA_DIR/$m --recursive");
