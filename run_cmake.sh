#! /usr/bin/env bash

mkdir -p build
cd build
cmake -DBLAS=Open -DCPU_ONLY=ON \
    -DPIPELINE_PATH=`realpath ../../pipeline_old` \
    -DDEEPLOCALIZER_CLASSIFIER_PATH=`realpath ../../deeplocalizer_classifier_old` \
    ..