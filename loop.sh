#!/bin/bash
PROJECT_DIR=$(readlink -f $(dirname $0))
cd $(dirname $0)
# . /opt/intel/openvino/bin/setupvars.sh
. /home/vvf/coding/cv-env/bin/activate
if [ -z "$1"  ]
then
python $PROJECT_DIR/license_ocr.py >> license_ocr.log 2>?1 &
OCR_PID=$!
fi
while true
do
    echo ""
    if [ -z "$1" ]
    then
      $PROJECT_DIR/watcher.py 2>> errors.log
    else
      $1 2>>errors_$2.log
    fi
    date >> errors.log
    echo ""
    echo restart  @ $(date)
    rm current?_*.mp4

    if [ -z "$1"  ]
    then
      if kill -0 $OCR_PID 2>/dev/null
      then
        echo "OCR already worked, PID=$OCR_PID"
      else
        python $PROJECT_DIR/license_ocr.py >> license_ocr.log &
        OCR_PID=$!
      fi
    fi
    
done

