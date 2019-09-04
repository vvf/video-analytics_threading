#!/bin/bash
PROJECT_DIR=$(readlink -f $(dirname $0))
cd $(dirname $0)
. /home/vvf/coding/cv-env/bin/activate
python $PROJECT_DIR/license_ocr.py >> license_ocr.log &
while true
do
    echo ""
    $PROJECT_DIR/watcher.py 2>> errors.log
    date >> errors.log
    echo ""
    echo restart  @ $(date)
    rm current_*.mp4
done

