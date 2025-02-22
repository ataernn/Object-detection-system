@echo off
powershell -Command "wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
powershell -Command "tar -zxf ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
pause
