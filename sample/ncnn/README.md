## NCNN 
* https://github.com/Tencent/ncnn
## Compile
* g++ -o yoloface-500k yoloface-500k_ncnn.cpp -I include/ncnn/ lib/libncnn.a `pkg-config --libs --cflags opencv` -fopenmp
