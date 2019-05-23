TensorRT Python Sample for Object Detection
======================================
TBD
</br>
</br>

## Install dependencies

```C
$ sudo apt-get install python3-pip
$ sudo pip3 install numpy pycuda
```

</br>
</br>

## Download model

Please download the object detection model from <a href=https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md>TensorFlow model zoo</a>.
</br>

```C
$ git https://github.com/AastaNV/TRT_object_detection.git
$ cd TRT_object_detection
$ cp [model].tar.gz model/
$ tar zxvf model/[model].tar.gz -C model/
```

##### Supported models:

- ssd_inception_v2_coco_2017_11_17
- ssd_mobilenet_v1_coco
- ssd_mobilenet_v2_coco

We will keep adding new model into our supported list.

</br>
</br>

## Update graphsurgeon converter

Edit /usr/lib/python3.6/dist-packages/graphsurgeon/node_manipulation.py

```C
diff --git a/node_manipulation.py b/node_manipulation.py
index d2d012a..1ef30a0 100644
--- a/node_manipulation.py
+++ b/node_manipulation.py
@@ -30,6 +30,7 @@ def create_node(name, op=None, _do_suffix=False, **kwargs):
     node = NodeDef()
     node.name = name
     node.op = op if op else name
+    node.attr["dtype"].type = 1
     for key, val in kwargs.items():
         if key == "dtype":
             node.attr["dtype"].type = val.as_datatype_enum
```
</br>
</br>

## RUN

**1. Maximize the Nano performance**
```C
sudo nvpmodel -m 0
sudo jetson_clocks
```
</br>

**2. Update main.py based on the model you used**
```C
from config import model_ssd_inception_v2_coco_2017_11_17 as model
from config import model_ssd_mobilenet_v1_coco_2018_01_28 as model
from config import model_ssd_mobilenet_v2_coco_2018_03_29 as model
```
</br>

**3. Execute**
```C
python3 main.py
```
</br>
</br>
</br>
</br>
</br>
