#!/bin/bash

gpu_id=$1
ext=$2

CUDA_VISIBLE_DEVICES=${gpu_id} py_gxdai run.py  --learning_rate "0.01" --momentum "0.9" --net_type "metricOnly" --margin_hid "50" --logfile "mapMetric.txt"

#CUDA_VISIBLE_DEVICES=${gpu_id} py_gxdai run.py.gxdai  --learning_rate "0.01" --momentum "0.9"
