#! /usr/bin/env bash

set -e

TEST_PATHFILE="test_preprocess_pathfile.txt"
TEST_IMG=$(realpath ../testdata/Cam_0_20140804152006_3.jpeg)
WITH_BORDER="/tmp/preprocess/Cam_0_20140804152006_3_wb.jpeg"
OUTPUT_PATHFILE="preprocess_pathfile.txt"
rm -f $WITH_BORDER
echo "Does $TEST_IMG exists?"
test -e $TEST_IMG

cd ../../source/tagger
echo $TEST_IMG > $TEST_PATHFILE
echo "Given a file of image path"
./preprocess -o /tmp/preprocess --output-pathfile $OUTPUT_PATHFILE $TEST_PATHFILE
echo "Then ./preprocess will add a border to the image"
test -e $WITH_BORDER

echo "Then ./preprocess will create an output pathfile"
test -e $OUTPUT_PATHFILE

echo "The output pathfile contains the path to the image with the new border"
test "`cat $OUTPUT_PATHFILE`" == "${WITH_BORDER}"

rm -f $TEST_PATHFILE $WITH_BORDER $OUTPUT_PATHFILE