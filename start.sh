#!/bin/bash
cd $(dirname $0)
screen -d -m -S car_watcher /home/vvf/coding/cars/loop.sh
