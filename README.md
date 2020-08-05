![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/7a96026f319ad31e28bc55458ee97e97.gif)
## ***Darknet Group convolution is not well supported on some GPUs such as NVIDIA PASCAL!!! 
* https://github.com/AlexeyAB/darknet/issues/6091#issuecomment-651667469
## MobileNetV2-YOLOv3-Lite&Nano Darknet
#### Mobile inference frameworks benchmark (4*ARM_CPU)
Network|VOC mAP(0.5)|COCO mAP(0.5)|Resolution|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
[MobileNetV2-YOLOv3-Lite](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Lite)(our)|73.26|37.44|320|28.42 ms|18 ms|1.8BFlops|8.0MB
[MobileNetV2-YOLOv3-Nano](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLOv3-Nano)(our)|65.27|30.13|320|10.16 ms|5 ms|0.5BFlops|3.0MB
[MobileNetV2-YOLOv3](https://github.com/eric612/MobileNet-YOLO)|70.7|&|352|32.15 ms|& ms|2.44BFlops|14.4MB
[MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)|72.7|&|300|26.37 ms|& ms|& BFlops|23.1MB
[YOLOv3-Tiny-Prn](https://github.com/AlexeyAB/darknet#pre-trained-models)|&|33.1|416|36.6 ms|& ms|3.5BFlops|18.8MB
[YOLOv4-Tiny](https://github.com/AlexeyAB/darknet#pre-trained-models)|&|40.2|416|44.6 ms|& ms|6.9BFlops|23.1MB
[YOLO-Nano](https://github.com/liux0614/yolo_nano)|69.1|&|416|& ms|& ms|4.57BFlops|4.0MB
* Support mobile inference frameworks such as NCNN&MNN
* The mnn benchmark only includes the forward inference time
* The ncnn benchmark is the forward inference time + post-processing time(NMS...) of the convolution feature map. 
* Darknet Train Configuration: CUDA-version: 10010 (10020), cuDNN: 7.6.4,OpenCV version: 4 GPU:RTX2080ti
## MobileNetV2-YOLOv3-Lite-COCO Test results
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/predictions.jpg)
## MobileNetV2-YOLO-Fastest
Network|Resolution|VOC mAP(0.5)|Inference time (DarkNet/i7-6700)|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
[MobileNetV2-YOLOv3-Fastest](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLO-Fastest/v1)|320|46.55|26 ms|8.2 ms|2.4 ms|0.13BFlops|700KB
[MobileNetV2-YOLOv3-Fastest-v2](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/MobileNetV2-YOLO-Fastest/v2)|320|50.13|27 ms|& ms|& ms|0.14BFlops|820KB
* 都2.4ms了，要啥mAP:sunglasses:
* V2 does not support MNN temporarily
* Suitable for hardware with extremely tight computing resources
* The mnn benchmark only includes the forward inference time
* The ncnn benchmark is the forward inference time + post-processing time(NMS...) of the convolution feature map. 
* This model is recommended to do some simple single object detection suitable for simple application scenarios
## MobileNetV2-YOLO-Fastest Test results
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/Fastest.jpg)
## yoloface-500k: 500kb yolo-Face-Detection
Network|Resolution|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:
UltraFace-version-RFB|320x240|&ms|3.36ms|0.1BFlops|1.3MB
UltraFace-version-Slim|320x240|&ms|3.06ms|0.1BFlops|1.2MB
[yoloface-500k](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/yoloface-500k/v1)|320x256|5.5ms|2.4ms|0.1BFlops|0.52MB
[yoloface-500k-v2](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/yoloface-500k/v2)|352x288|4.7ms|&ms|0.1BFlops|0.42MB
* 都500k了，要啥mAP:sunglasses:
* Inference time (DarkNet/i7-6700):13ms
* The mnn benchmark only includes the forward inference time
* The ncnn benchmark is the forward inference time + post-processing time(NMS...) of the convolution feature map. 
## Wider Face Val
Model|Easy Set|Medium Set|Hard Set
------|--------|----------|--------
libfacedetection v1（caffe）|0.65 |0.5       |0.233
libfacedetection v2（caffe）|0.714 |0.585       |0.306
Retinaface-Mobilenet-0.25 (Mxnet)   |0.745|0.553|0.232
version-slim-320|0.77     |0.671       |0.395
version-RFB-320|0.787     |0.698       |0.438
[yoloface-500k-320](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/yoloface-500k/v1)|**0.728**|**0.682**|**0.431**|
[yoloface-500k-352-v2](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/yoloface-500k/v2)|**0.768**|**0.729**|**0.490**|
* yoloface-500k-v2：The SE&CSP module is added
* V2 does not support MNN temporarily
* wider_face_val(ap05): yoloface-500k: 53.75 yoloface-500k-v2: 56.69
## YoloFace-500k Test results(thresh 0.7)
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/p1.jpg)
## YoloFace-500k-v2 Test results(thresh 0.7)
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/f2.jpg)
## yoloface-100k: Sub-millisecond face detection model
Network|Resolution|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|Inference time (DarkNet/i7-6700)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:
[yoloface-100k](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/yoloface-100k)|112x112|0.8ms|0.325ms|2ms|0.009BFlops|109kb
* ***For the close-range face detection model in a specific scene, the recommended detection distance is 1.5m***
* The detection distance can be increased by increasing the model input size (multiple of 16)
## YoloFace-100k Test results(thresh 0.7 input_size:112x112)
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/yoloface-100k-1.jpg)
## YoloFace-100k Test results(thresh 0.7 input_size:224x224)
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/yoloface-100k-2.jpg)
## YoloFace100k-landmark106(Ultra lightweight 106 point face-landmark model)
Network|Resolution|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|Weight size
:---:|:---:|:---:|:---:|:---:
[landmark106](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/yoloface100k-landmark106)|112x112|0.6ms|0.5ms|1.4MB
* Face detection: yoloface-100k Landmark: landmark106
## YoloFace100k-landmark106 Test results
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/yoloface100k-landmark106/yoloface-100k-landmark106.jpg)
## Reference&Framework instructions&How to Train
* https://github.com/AlexeyAB/darknet
* You must use a pre-trained model to train your own data set. You can make a pre-trained model based on the weights of COCO training in this project to initialize the network parameters
* 交流qq群:1062122604
## About model selection
* MobileNetV2-YOLOv3-SPP:  Nvidia Jeston, Intel Movidius, TensorRT，NPU，OPENVINO...High-performance embedded side
* MobileNetV2-YOLOv3-Lite: High Performance ARM-CPU，Qualcomm Adreno GPU， ARM82...High-performance mobile
* MobileNetV2-YOLOv3-NANO： ARM-CPU...Computing resources are limited
* MobileNetV2-YOLOv3-Fastest： ....... Can you do personal face detection???It’s better than nothing
## NCNN conversion tutorial
* Benchmark:https://github.com/Tencent/ncnn/tree/master/benchmark
* NCNN supports direct conversion of darknet models
* darknet2ncnn: https://github.com/Tencent/ncnn/tree/master/tools/darknet
## NCNN C++ Sample
* https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/tree/master/sample/ncnn
## NCNN Android Sample
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/MobileNetV2-YOLOV3-Nano.gif)
* https://github.com/dog-qiuqiu/Android_MobileNetV2-YOLOV3-Nano-NCNN
* APK:https://github.com/dog-qiuqiu/Android_MobileNetV2-YOLOV3-Nano-NCNN/blob/master/app/release/MobileNetv2-yolov3-nano.apk
## DarkNet2Caffe tutorial
### Environmental requirements

* Python2.7
* python-opencv
* Caffe(add upsample layer https://github.com/dog-qiuqiu/caffe)
* You have to compile cpu version of caffe！！！
  ```
	cd darknet2caffe/
	python darknet2caffe.py MobileNetV2-YOLOv3-Nano-voc.cfg MobileNetV2-YOLOv3-Nano-voc.weights MobileNetV2-YOLOv3-Nano-voc.prototxt MobileNetV2-YOLOv3-Nano-voc.caffemodel
	cp MobileNetV2-YOLOv3-Nano-voc.prototxt sample
	cp MobileNetV2-YOLOv3-Nano-voc.caffemodel sample
	cd sample
	python detector.py
  ```
### MNN conversion tutorial
* Benchmark:https://www.yuque.com/mnn/cn/tool_benchmark
* Convert darknet model to caffemodel through darknet2caffe
* Manually replace the upsample layer in prototxt with the interp layer
* Take the modification of MobileNetV2-YOLOv3-Nano-voc.prototxt as an example
```
	#layer {
	#    bottom: "layer71-route"
	#    top: "layer72-upsample"
	#    name: "layer72-upsample"
	#    type: "Upsample"
	#    upsample_param {
	#        scale: 2
	#    }
	#}
	layer {
	    bottom: "layer71-route"
	    top: "layer72-upsample"
	    name: "layer72-upsample"
	    type: "Interp"
	    interp_param {
		height:20  #upsample h size
		width:20   #upsample w size
	    }
	}

```
* MNN conversion: https://www.yuque.com/mnn/cn/model_convert
## Thanks
* https://github.com/shicai/MobileNet-Caffe
* https://github.com/WZTENG/YOLOv5_NCNN 
* https://github.com/AlexeyAB/darknet
* https://github.com/Tencent/ncnn
* https://gluon-cv.mxnet.io/
