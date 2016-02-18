#! /usr/bin/env bash

set -e
set -o xtrace

TEST_PATHFILE="test_preprocess_pathfile.txt"
IMG="Cam_2_20150828143300_888543_wb.jpeg"
TEST_IMG=$(realpath "../testdata/$IMG")
WITH_BORDER="/tmp/preprocess/${IMG%.*}_wb.jpeg"
OUTPUT_PATHFILE="preprocess_pathfile.txt"
TMP_DIR="/tmp/preprocess"
rm -f $WITH_BORDER
echo "Does $TEST_IMG exists?"
test -e $TEST_IMG

cd ../../source/tagger
echo $TEST_IMG > $TEST_PATHFILE
echo "Given a file of image path"

./preprocess -o $TMP_DIR --output-pathfile $OUTPUT_PATHFILE $TEST_PATHFILE
echo "Then ./preprocess will add a border to the image"
if [ ! -e "$WITH_BORDER" ]; then
    echo "WITH_BORDER does not exists: $WITH_BORDER"
    false
fi

echo "Then ./preprocess will create an output pathfile"

if [ ! -e "$TMP_DIR/$OUTPUT_PATHFILE" ]; then
    echo "OUTPUT_PATHFILE does not exists: $TMP_DIR/$OUTPUT_PATHFILE"
    false
fi
echo "The output pathfile contains the path to the image with the new border"
if [ "`cat $TMP_DIR/$OUTPUT_PATHFILE`" != "${WITH_BORDER}" ]; then
    echo "output pathfile has wrong content"
    echo "got:"
    cat $TMP_DIR/$OUTPUT_PATHFILE
    echo "expected:"
    echo "$WITH_BORDER"
    false
fi

# rm -f $TEST_PATHFILE $WITH_BORDER $OUTPUT_PATHFILE