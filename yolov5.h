#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// #include "common.h"
#include "tengine/c_api.h"
// #include "tengine_operations.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class YOLOv5Detector
{
    public:
        YOLOv5Detector();
        ~YOLOv5Detector();
        int init(std::string modelpath, int rows, int cols, int nthreads, float confidence, float nms);
        void get_input_data_focus(cv::Mat image, float* input_data, int letterbox_rows, int letterbox_cols, const float* mean, const float* scale);
        void infer(cv::Mat image);
        void generate_proposals(int stride, const float* feat, float prob_threshold, std::vector<Object>& objects,
                               int letterbox_cols, int letterbox_rows);

    private:
        int m_letterbox_rows;
        int m_letterbox_cols;
        int m_num_thread;
        std::string m_model_path;
        float m_prob_threshold;
        float m_nms_threshold;
        
        graph_t graph;
        int img_c = 3;
        const float mean[3] = {0, 0, 0};
        const float scale[3] = {0.003921, 0.003921, 0.003921};
        int img_size;
        std::vector<float> input_data;
};