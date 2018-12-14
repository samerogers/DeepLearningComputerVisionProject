#!/bin/bash

cd TinyTLP

for D in *; do
    if [ -d "${D}" ]; then
        echo "${D}"   # your processing here
        rm -r ${D}/groundtruth_rect.txt
        mv ${D}/groundtruth_rect1.txt ${D}/groundtruth_rect.txt
    fi
done
