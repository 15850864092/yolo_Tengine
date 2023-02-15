#include "yolov5.h"

YOLOv5Detector::YOLOv5Detector()
{

}

YOLOv5Detector::~YOLOv5Detector()
{
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}

int check_file_exist(const char* file_name)
{
    FILE* fp = fopen(file_name, "r");
    if (!fp)
    {
        fprintf(stderr, "Input file not existed: %s\n", file_name);
        return 0;
    }
    fclose(fp);
    return 1;
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"};

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }

    cv::imwrite("yolov5m6_out.jpg", image);
}

void YOLOv5Detector::generate_proposals(int stride, const float* feat, float prob_threshold, std::vector<Object>& objects,
                               int letterbox_cols, int letterbox_rows)
{
    // static float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
    static float anchors[24] = {19,27, 44,40, 38,94, 96,68, 86,152, 180,137, 140,301, 303,264, 238,542, 436,615, 739,380, 925,792};

    int anchor_num = 3;
    int feat_w = letterbox_cols / stride;
    int feat_h = letterbox_rows / stride;
    int cls_num = 80;
    int anchor_group;
    if (stride == 8)
        anchor_group = 1;
    if (stride == 16)
        anchor_group = 2;
    if (stride == 32)
        anchor_group = 3;
    if (stride == 64)
        anchor_group = 4;
    for (int h = 0; h <= feat_h - 1; h++)
    {
        for (int w = 0; w <= feat_w - 1; w++)
        {
            for (int a = 0; a <= anchor_num - 1; a++)
            {
                //process cls score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s <= cls_num - 1; s++)
                {
                    float score = feat[a * feat_w * feat_h * (cls_num + 5) + h * feat_w * (cls_num + 5) + w * (cls_num + 5) + s + 5];
                    if (score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }
                //process box score
                float box_score = feat[a * feat_w * feat_h * (cls_num + 5) + (h * feat_w) * (cls_num + 5) + w * (cls_num + 5) + 4];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if (final_score >= prob_threshold)
                {
                    int loc_idx = a * feat_h * feat_w * (cls_num + 5) + h * feat_w * (cls_num + 5) + w * (cls_num + 5);
                    float dx = sigmoid(feat[loc_idx + 0]);
                    float dy = sigmoid(feat[loc_idx + 1]);
                    float dw = sigmoid(feat[loc_idx + 2]);
                    float dh = sigmoid(feat[loc_idx + 3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj);
                }
            }
        }
    }
}

int YOLOv5Detector::init(std::string modelpath, int rows, int cols, int nthreads, float confidence, float nms)
{
    m_letterbox_rows = rows;
    m_letterbox_cols = cols;
    m_model_path = modelpath;
    m_num_thread = nthreads;
    m_prob_threshold = confidence;
    m_nms_threshold = nms;

    if (!check_file_exist(m_model_path.c_str()))
        return -1;
    /* set runtime options */
    struct options opt;
    opt.num_thread = m_num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());
    /* create graph, load tengine model xxx.tmfile */
    graph = create_graph(nullptr, "tengine", m_model_path.c_str());
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        return -1;
    }
    img_size = m_letterbox_rows * m_letterbox_cols * img_c;
    int dims[] = {1, 12, int(m_letterbox_rows / 2), int(m_letterbox_cols / 2)};
    input_data.resize(img_size);
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }
    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }
    if (set_tensor_buffer(input_tensor, input_data.data(), img_size * sizeof(float)) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }
    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }
    printf("Tengine init success\n");
    return 0;
}

void YOLOv5Detector::get_input_data_focus(cv::Mat img, float* input_data, int letterbox_rows, int letterbox_cols, const float* mean, const float* scale)
{
    /* letterbox process to support different letterbox size */
    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((m_letterbox_rows * 1.0 / img.rows) < (m_letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = m_letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = m_letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    cv::resize(img, img, cv::Size(resize_cols, resize_rows));
    img.convertTo(img, CV_32FC3);
    // Generate a gray image for letterbox using opencv
    cv::Mat img_new(m_letterbox_cols, m_letterbox_rows, CV_32FC3, cv::Scalar(0.5 / scale[0] + mean[0], 0.5 / scale[1] + mean[1], 0.5 / scale[2] + mean[2]));
    int top = (m_letterbox_rows - resize_rows) / 2;
    int bot = (m_letterbox_rows - resize_rows + 1) / 2;
    int left = (m_letterbox_cols - resize_cols) / 2;
    int right = (m_letterbox_cols - resize_cols + 1) / 2;
    // Letterbox filling
    cv::copyMakeBorder(img, img_new, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    img_new.convertTo(img_new, CV_32FC3);
    float* img_data = (float*)img_new.data;
    std::vector<float> input_temp(3 * m_letterbox_cols * m_letterbox_rows);
    /* nhwc to nchw */
    for (int h = 0; h < m_letterbox_rows; h++)
    {
        for (int w = 0; w < m_letterbox_cols; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index = h * m_letterbox_cols * 3 + w * 3 + c;
                int out_index = c * m_letterbox_rows * m_letterbox_cols + h * m_letterbox_cols + w;
                input_temp[out_index] = (img_data[in_index] - mean[c]) * scale[c];
            }
        }
    }
    /* focus process */
    for (int i = 0; i < 2; i++) // corresponding to rows
    {
        for (int g = 0; g < 2; g++) // corresponding to cols
        {
            for (int c = 0; c < 3; c++)
            {
                for (int h = 0; h < m_letterbox_rows / 2; h++)
                {
                    for (int w = 0; w < m_letterbox_cols / 2; w++)
                    {
                        int in_index = i + g * m_letterbox_cols + c * m_letterbox_cols * m_letterbox_rows + h * 2 * m_letterbox_cols + w * 2;
                        int out_index = i * 2 * 3 * (m_letterbox_cols / 2) * (m_letterbox_rows / 2) + g * 3 * (m_letterbox_cols / 2) * (m_letterbox_rows / 2) + c * (m_letterbox_cols / 2) * (m_letterbox_rows / 2) + h * (m_letterbox_cols / 2) + w;

                        input_data[out_index] = input_temp[in_index];
                    }
                }
            }
        }
    }
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void YOLOv5Detector::infer(cv::Mat img)
{
    get_input_data_focus(img, input_data.data(), m_letterbox_rows, m_letterbox_cols, mean, scale);
    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "Run graph failed\n");
    }
    //postprocess
    // 0: 1, 3, 20, 20, 85
    // 1: 1, 3, 40, 40, 85
    // 2: 1, 3, 80, 80, 85
    // 3: 1, 3, 160, 160, 85
    tensor_t p8_output = get_graph_output_tensor(graph, 0, 0);
    tensor_t p16_output = get_graph_output_tensor(graph, 1, 0);
    tensor_t p32_output = get_graph_output_tensor(graph, 2, 0);
    tensor_t p64_output = get_graph_output_tensor(graph, 3, 0);

    float* p8_data = (float*)get_tensor_buffer(p8_output);
    float* p16_data = (float*)get_tensor_buffer(p16_output);
    float* p32_data = (float*)get_tensor_buffer(p32_output);
    float* p64_data = (float*)get_tensor_buffer(p64_output);

    std::vector<Object> proposals;
    std::vector<Object> objects8;
    std::vector<Object> objects16;
    std::vector<Object> objects32;
    std::vector<Object> objects64;
    std::vector<Object> objects;

    generate_proposals(64, p64_data, m_prob_threshold, objects64, m_letterbox_cols, m_letterbox_rows);
    proposals.insert(proposals.end(), objects64.begin(), objects64.end());
    generate_proposals(32, p32_data, m_prob_threshold, objects32, m_letterbox_cols, m_letterbox_rows);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    generate_proposals(16, p16_data, m_prob_threshold, objects16, m_letterbox_cols, m_letterbox_rows);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generate_proposals(8, p8_data, m_prob_threshold, objects8, m_letterbox_cols, m_letterbox_rows);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, m_nms_threshold);

    float scale_letterbox;
    int resize_rows;
    int resize_cols;
    if ((m_letterbox_rows * 1.0 / img.rows) < (m_letterbox_cols * 1.0 / img.cols))
    {
        scale_letterbox = m_letterbox_rows * 1.0 / img.rows;
    }
    else
    {
        scale_letterbox = m_letterbox_cols * 1.0 / img.cols;
    }
    resize_cols = int(scale_letterbox * img.cols);
    resize_rows = int(scale_letterbox * img.rows);

    int tmp_h = (m_letterbox_rows - resize_rows) / 2;
    int tmp_w = (m_letterbox_cols - resize_cols) / 2;

    float ratio_x = (float)img.rows / resize_rows;
    float ratio_y = (float)img.cols / resize_cols;

    int count = picked.size();
    fprintf(stderr, "detection num: %d\n", count);

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x);
        float y0 = (objects[i].rect.y);
        float x1 = (objects[i].rect.x + objects[i].rect.width);
        float y1 = (objects[i].rect.y + objects[i].rect.height);

        x0 = (x0 - tmp_w) * ratio_x;
        y0 = (y0 - tmp_h) * ratio_y;
        x1 = (x1 - tmp_w) * ratio_x;
        y1 = (y1 - tmp_h) * ratio_y;

        x0 = std::max(std::min(x0, (float)(img.cols - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img.rows - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img.cols - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img.rows - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    draw_objects(img, objects);
}