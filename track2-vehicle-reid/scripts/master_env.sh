#!/usr/bin/env bash
# Master environment setting

# Get GPUID
if [ -z $1 ]; then
    GPUID=0
else
    GPUID=$1
fi

# Get No WORKERS
N_WORKERS="8"
