#!/bin/bash

# Doing this we unzip the graphs downloaded but still present as zip files (however not with the
# extension .zip).
# https://stackoverflow.com/questions/14813130/find-all-zips-and-unzip-in-place-unix

# do this cleanup once with the downloaded instances!

cd "${1}"

# let's unzip everything there is
unzip -o "*"

# replace "," with " " in all .edges files
# yeeep.. this takes a while, be patient...
sed -i 's/[,]/ /g' *.edges

# HACK: rename the ones from SNAP first!
mv wiki-Talk wiki-Talk.edges
mv cit-Patents cit-Patents.edges
mv roadNet-CA roadNet-CA.edges
mv amazon0601 amazon0601.edges
mv web-Google web-Google.edges
mv web-BerkStan web-BerkStan.edges
mv wiki-topcats  wiki-topcats .edges
mv soc-LiveJournal1 soc-LiveJournal1.edges
mv wiki-talk-temporal wiki-talk-temporal.edges

# rename all files with ending .edges to no endings (we need this for simex to read in instances)
find . -name "*.edges" -exec sh -c 'f="{}"; mv -- "$f" "${f%.edges}"' ';'
