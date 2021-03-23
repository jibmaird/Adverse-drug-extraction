#!/usr/bin/perl -w

use strict;

use vars qw ();

#Process train sentences (extract sentences and their annotations)
#system("./train_sentences.pl"); It requires python37

#system("./identify_errors.pl");

#system("./corenlp_server.pl");
#system("./align.pl");
system("./align_from_checkpoint.pl");

#system("./top_matches.pl");
