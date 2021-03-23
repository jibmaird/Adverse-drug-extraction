#!/usr/bin/perl -w

use strict;

#Upload NERdata 
#my $data_d = "/home/jibmaird/Data/Corpora/ADE-v2/Exper/Biobert/bio";
#my $ner_d = "biobert/NERdata/ADE";

#opendir(D,"$data_d")||die;
#my @File = grep /^[^\.]/,readdir D;
#closedir(D);
#foreach my $f (@File) {
#    system("aws --endpoint-url=http://s3.ap-geo.objectstorage.softlayer.net s3 cp $data_d\/$f s3://dmi-bucket\/$ner_d/$f");
#}

# Upload pretrained models
my $data_d = "/home/jibmaird/Projects";
#my $pretrain_d = "biobert-pretrained/biobert_v1.0_pubmed";
my $pretrain_d = "biobert-pretrained/biobert_v1.0_pubmed_pmc";
opendir(D,"$data_d\/$pretrain_d")||die;
my @File = grep /\w/,readdir D;
closedir(D);
foreach my $f (@File) {
    system("aws --endpoint-url=http://s3.ap-geo.objectstorage.softlayer.net s3 cp $data_d\/$pretrain_d\/$f s3://dmi-bucket\/$pretrain_d/$f");
}
