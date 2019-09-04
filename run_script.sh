#!/usr/bin/env bash

cd $(dirname $0)
export WSSERVER_HOST=0.0.0.0
. /home/vvf/coding/cv-env/bin/activate
if [ "$1" == "loop" ]
then
    shift
    while true
    do
        python $*
        echo restart $*
    done
else
    python $*
fi