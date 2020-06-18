## MobileNetv2-YOLOv3-SPP Darknet 

A darknet implementation of MobileNetv2-YOLOv3-SPP detection network

Network|VOC mAP(0.5)|Resolution|Inference Time(GTX2080ti)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3-SPP|71.7|416|5ms|5.5BFlops|14.2
## Mobile inference frameworks benchmark
Network|VOC mAP(0.5)|COCO mAP(0.5)|Resolution|Inference time (NCNN/Mate30)|Inference Time(MNN/Mate30)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3-Lite|65.1|35.2|320|33 ms|&ms|2.1BFlops|9.8MB
[yolov3-tiny-prn](https://github.com/AlexeyAB/darknet#pre-trained-models)|&|33.1|416|& ms|&ms|3.5BFlops|18.8MB

* Darknet Configuration: CUDA-version: 10010 (10020), cuDNN: 7.6.4,OpenCV version: 4.9.1
* Darknet Packet convolution is not well supported on some GPUs such as gtx1080ti, and the inference time is 100ms
* Provide model conversion scripts to transplant models to inference frameworks such as caffe
* Support mobile inference frameworks such as NCNN/MNN 
## 正在完成中...
## Reference&Framework instructions
https://github.com/AlexeyAB/darknet
