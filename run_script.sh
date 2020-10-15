#!/usr/bin/env bash

cd $(dirname $0)
export WSSERVER_HOST=0.0.0.0
. /home/vvf/coding/cv-env/bin/activate
. /opt/intel/openvino/bin/setupvars.sh
if [ "$1" == "loop" ]
then
    shift
    while true
    do
        python $@
        echo restart $@
    done
else
    if [ "$1" == "i" ]
    then
        ipython
    else
        echo python $@
        python $@
    fi
fi