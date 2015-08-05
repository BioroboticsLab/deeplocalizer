#!/usr/bin/env bash

set -e
set -o xtrace

TEST_PATHFILE="test_pathfile.txt"
TRAIN_PATHFILE="train_pathfile.txt"

TEST_IMG=$(realpath "../testdata/Cam_2_20140805145841_2_wb.jpeg")
IMG_TAGGER_DESC=$(realpath "../testdata/Cam_2_20140805145841_2_wb.jpeg.tagger.desc")

DATA_DIR=$(mktemp -d)
echo "Does $TEST_IMG exists?"
test -e $TEST_IMG

cd ../../source/tagger
echo `pwd`
echo $TEST_IMG > $TEST_PATHFILE
echo $TEST_IMG > $TRAIN_PATHFILE
mkdir -p /tmp/data

test -e $IMG_TAGGER_DESC

cat $TEST_PATHFILE
cat $TRAIN_PATHFILE
./generate_dataset --test ${TEST_PATHFILE} --train ${TRAIN_PATHFILE} -f images -o ${DATA_DIR}
echo "Given a test and train pathfile then ./generate_dataset will generate an train and test dataset"
test -e ${DATA_DIR}/train
test -e ${DATA_DIR}/test

rm -rf $DATA_DIR

