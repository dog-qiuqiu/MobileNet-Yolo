## MobileNetv2-YOLOv3-SPP Darknet 

A darknet implementation of MobileNetv2-YOLOv3-SPP detection network

Network|VOC mAP(0.5)|Resolution|Inference Time(GTX2080ti)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3-SPP|71.7|416|5ms|5.5BFlops|14.2

*emmmm...这个懒得训练，mAP就凑合这样吧
## Mobile inference frameworks benchmark (4*ARM_CPU)
Network|VOC mAP(0.5)|COCO mAP(0.5)|Resolution|Inference time (NCNN/Kirin 990)|Inference time (MNN arm82/Kirin 990)|FLOPS|Weight size
:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
MobileNetV2-YOLOv3-Lite|72.61|35.2|320|33 ms|18 ms|2.1BFlops|9.8MB
MobileNetV2-YOLO-Tiny|61.17|30.4|304|26 ms|11 ms|1.5Flops|3.9MB
[yolov3-tiny-prn](https://github.com/AlexeyAB/darknet#pre-trained-models)|&|33.1|416|& ms|& ms|3.5BFlops|18.8MB


* Darknet Train Configuration: CUDA-version: 10010 (10020), cuDNN: 7.6.4,OpenCV version: 4 GPU:RTX2080ti
* Darknet Packet convolution is not well supported on some GPUs such as gtx1080ti, and the MobileNetV2-YOLOv3-SPP	inference time is 100ms
* Support mobile inference frameworks such as NCNN&MNN 
## MobileNetV2-YOLOv3-Lite-COCO Test results
![image](https://github.com/dog-qiuqiu/MobileNetv2-YOLOV3/blob/master/data/predictions.jpg)


## NCNN conversion tutorial
* benchmark:https://github.com/Tencent/ncnn/tree/master/benchmark
* darknet2ncnn: https://github.com/Tencent/ncnn/tree/master/tools/darknet
## NCNN Android Sample
* 白嫖中....
## MNN conversion tutorial
* 待完成
## Reference&Framework instructions
https://github.com/AlexeyAB/darknet
