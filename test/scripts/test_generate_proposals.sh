#! /usr/bin/env bash

set -e
set -o xtrace

TEST_PATHFILE="test_pathfile.txt"
TEST_IMG=$(find `pwd`/../testdata -name with_one_tag.jpeg | xargs realpath)
TEST_IMG_DESC="${TEST_IMG}.proposal.json"
rm -f $TEST_IMG_DESC
echo "Does $TEST_IMG exists?"
test -e $TEST_IMG

cd ../../source/tagger
echo `pwd`
echo $TEST_IMG > $TEST_PATHFILE
echo "Given a file of image path"
./generate_proposals $TEST_PATHFILE
echo "Then ./generate_proposals will generate .json files for every image"
test -e ${TEST_IMG_DESC}
rm -f $TEST_IMG_DESC