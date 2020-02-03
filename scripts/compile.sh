#!/bin/bash
echo 'Compiling NMS module...'
(cd lib/nms; python2 setup_linux.py build_ext --inplace)
echo 'Compiling bbox module...'
(cd lib/bbox; python2 setup_linux.py build_ext --inplace)
echo 'Compiling chips module...'
(cd lib/chips; python2 setup.py)
echo 'Compiling coco api...'
(cd lib/dataset/pycocotools; python2 setup_linux.py build_ext --inplace)
echo 'All Done!'