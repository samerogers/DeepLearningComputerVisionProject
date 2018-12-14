#!/bin/bash
cd ./data

for D in *; do
    if [ -d "${D}" ]; then
        echo "${D}"   # your processing here
        rm -r ${D}/output
    fi
done
