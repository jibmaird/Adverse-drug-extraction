#!/usr/bin/perl -w

$_ = `bx ml show training-runs $ARGV[0]`;
m /.*?Model location\s*(.*?)\s*\n/;
print "$1#\n";
system("aws --endpoint-url=http://s3.ap-geo.objectstorage.softlayer.net s3 cp s3://dmi-bucket/$1/training\-log.txt training\-log.txt");
