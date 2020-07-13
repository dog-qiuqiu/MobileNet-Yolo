#include "net.h"
#include "platform.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN


static int detect_yolov3(const cv::Mat& bgr)
{
    ncnn::Net yolov3;
    cv::Mat image = bgr.clone();

#if NCNN_VULKAN
    yolov3.opt.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    yolov3.load_param("model/yoloface-500k.param");
    yolov3.load_model("model/yoloface-500k.bin");

    const int target_size_width = 320;
    const int target_size_height = 256;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, target_size_width, target_size_height);

    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = yolov3.create_extractor();
    ex.set_num_threads(4);
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("output", out);

    for (int i = 0; i < out.h; i++)
    {
        float x1, y1, x2, y2, score, label;
        const float* values = out.row(i);
        
        x1 = values[2] * img_w;
        y1 = values[3] * img_h;
        x2 = values[4] * img_w;
        y2 = values[5] * img_h;
        score = values[1];
        label = values[0];
        
        cv::rectangle (image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 255), 2, 8, 0);
    }
    cv::imshow("image", image);
    cv::waitKey(0);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];
    cv::Mat m = cv::imread(imagepath, 1);

    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    detect_yolov3(m);
    return 0;
}
