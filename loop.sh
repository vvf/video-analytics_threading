#!/bin/bash
PROJECT_DIR=$(readlink -f $(dirname $0))
cd $(dirname $0)
. /home/vvf/coding/cv-env/bin/activate
python $PROJECT_DIR/license_ocr.py >> license_ocr.log &
OCR_PID=$!
while true
do
    echo ""
    $PROJECT_DIR/watcher.py 2>> errors.log
    date >> errors.log
    echo ""
    echo restart  @ $(date)
    rm current_*.mp4

    kill -0 $OCR_PID 2>/dev/null
    python $PROJECT_DIR/license_ocr.py >> license_ocr.log &
    OCR_PID=$!
done

