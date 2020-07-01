## 路过的大佬给颗星星吧
## 待办
* 测试CspNet，不好使…不用了
* 500kb的yolo-Face-Detection
* 开放andooid示例项目(求大佬)
## MobileNetv2-YOLOv3-SPP Darknet 

A darknet implementation of MobileNetv2-YOLOv3-SPP detection network

Network|COCO mAP(0.5)|Resolution|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3-SPP|42.6|416|6.1BFlops|17.6MB
[YOLOv4-Tiny](https://github.com/AlexeyAB/darknet#pre-trained-models)|40.2|416|6.9BFlops|23.1MB

*emmmm...这个懒得训练，mAP就凑合这样吧
## ***Darknet Group convolution is not well supported on some GPUs such as NVIDIA PASCAL!!! The MobileNetV2-YOLOv3-SPP	inference time is 100ms at GTX1080ti, but RTX2080 inference time is 5ms!!!***
* https://github.com/AlexeyAB/darknet/issues/6091#issuecomment-651667469
## MobileNetV2-YOLOv3-Lite&Nano Darknet
#### Mobile inference frameworks benchmark (4*ARM_CPU)
Network|VOC mAP(0.5)|COCO mAP(0.5)|Resolution|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3-Lite|72.61|36.57|320|31.58 ms|18 ms|1.8BFlops|8.0MB
MobileNetV2-YOLOv3-Nano|65.27|30.13|320|13 ms|5 ms|0.5BFlops|3.0MB
[YOLOv3-Tiny-Prn](https://github.com/AlexeyAB/darknet#pre-trained-models)|&|33.1|416|36.6 ms|& ms|3.5BFlops|18.8MB
[YOLO-Nano](https://github.com/liux0614/yolo_nano)|69.1|&|416|& ms|& ms|4.57BFlops|4.0MB
* Darknet Train Configuration: CUDA-version: 10010 (10020), cuDNN: 7.6.4,OpenCV version: 4 GPU:RTX2080ti
* Support mobile inference frameworks such as NCNN&MNN
## MobileNetV2-YOLOv3-Lite-COCO Test results
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/predictions.jpg)
## MobileNetV2-YOLO-Fastest
Network|Resolution|VOC mAP(0.5)|Inference time (DarkNet/i7-6700)|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3-Fastest|320|46.55|26 ms|8.2 ms|2.4 ms|0.13BFlops|700KB
* 快就完了，要啥mAP:sunglasses:
* Suitable for hardware with extremely tight computing resources
* This model is recommended to do some simple single object detection suitable for simple application scenarios
## MobileNetV2-YOLO-Fastest Test results
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/Fastest.jpg)
## Reference&Framework instructions&How to Train
* https://github.com/AlexeyAB/darknet
* You must use a pre-trained model to train your own data set. You can make a pre-trained model based on the weights of COCO training in this project to initialize the network parameters
* 交流qq群:1062122604
## About model selection
* MobileNetV2-YOLOv3-SPP:  Nvidia Jeston, Intel Movidius, TensorRT，NPU，OPENVINO...High-performance embedded side
* MobileNetV2-YOLOv3-Lite: High Performance ARM-CPU，Qualcomm Adreno GPU， ARM82...High-performance mobile
* MobileNetV2-YOLOv3-NANO： ARM-CPU...Computing resources are limited
* MobileNetV2-YOLOv3-Fastest： ....... Can you do personal face detection???It’s better than nothing
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
		height:20 
		      width:20
	    }
	}

```
* MNN conversion: https://www.yuque.com/mnn/cn/model_convert
## NCNN conversion tutorial
* NCNN supports direct conversion of darknet models
* benchmark:https://github.com/Tencent/ncnn/tree/master/benchmark
* darknet2ncnn: https://github.com/Tencent/ncnn/tree/master/tools/darknet
## NCNN Android Sample
* 白嫖中....
## Thanks
* https://github.com/shicai/MobileNet-Caffe
* https://github.com/AlexeyAB/darknet
* https://github.com/Tencent/ncnn
* https://gluon-cv.mxnet.io/
