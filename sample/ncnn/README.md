## NCNN 
* https://github.com/Tencent/ncnn
## Compile
* g++ -o yoloface yoloface-500k-lanmark106.cpp -I include/ncnn/ lib/libncnn.a `pkg-config --libs --cflags opencv` -fopenmp
* Usage: ./yoloface
