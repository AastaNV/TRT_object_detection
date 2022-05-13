TensorRT Python Sample for Object Detection
======================================

Performance includes memcpy and inference.
</br>

| Model | Input Size | TRT Nano |
|:------|:----------:|-----------:|
| ssd_inception_v2_coco(2017) | 300x300 | 49ms |
| ssd_mobilenet_v1_coco | 300x300 | 36ms |
| ssd_mobilenet_v2_coco | 300x300 | 46ms |

Since the optimization of preprocessing is not ready yet, we don't include image read/write time here.
</br>
</br>

## Install dependencies

```C
$ sudo apt-get install python3-pip libhdf5-serial-dev hdf5-tools
$ pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.5 --user
$ pip3 install numpy pycuda --user
```

</br>
</br>

## Download model

Please download the object detection model from <a href=https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md>TensorFlow model zoo</a>.
</br>

```C
$ git clone https://github.com/AastaNV/TRT_object_detection.git
$ cd TRT_object_detection
$ mkdir model
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
$ sudo nvpmodel -m 0
$ sudo jetson_clocks
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
$ python3 main.py [image]
```

It takes some time to compile a TensorRT model when the first launching.
</br>
After that, TensorRT engine can be created directly with the serialized .bin file
</br>
</br>
@ To get more memory, it's recommended to turn-off X-server.
</br>
</br>
</br>
</br>
</br>
