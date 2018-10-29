#!/bin/bash


ONNX_URL="https://github.com/Hakuyume/menoh-yolo/releases/download/assets/yolo_v2_voc0712.onnx"
CONF_URL="https://github.com/Hakuyume/menoh-yolo/releases/download/assets/yolo_v2_voc0712.json"

DATA_DIR="./test/data"

ONNX_PATH=$DATA_DIR/${ONNX_URL##*/}
CONF_PATH=$DATA_DIR/${CONF_URL##*/}


mkdir -p $DATA_DIR

if [ ! -f $ONNX_PATH ]; then
    echo Downloading ONNX file to $DATA_DIR ..
    cd $DATA_DIR
    curl -LO $ONNX_URL
    cd -
fi

if [ ! -f $CONF_PATH ]; then
    echo Downloading config file to $DATA_DIR ..
    cd $DATA_DIR
    curl -LO $CONF_URL
    cd -
fi

