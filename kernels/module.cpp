#include <torch/extension.h>

// forward declarations
torch::Tensor rgb_to_gray(torch::Tensor img);
torch::Tensor box_blur(torch::Tensor img, int blurSize);
torch::Tensor sobel(torch::Tensor img);
torch::Tensor dilation(torch::Tensor img, int filterSize);
torch::Tensor erosion(torch::Tensor img, int filterSize);
torch::Tensor adaptive_threshold_tensor(torch::Tensor input, double max_value, int method, int type, int blockSize, double C);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("rgb2gray", &rgb_to_gray, "rgb to grayscale kernel");
    m.def("blur", &box_blur, "box blur kernel");
    m.def("sobel", &sobel, "sobel filter kernel");
    m.def("dilate", &dilation, "dilation kernel");
    m.def("erode", &erosion, "erosion kernel");
    m.def("adaptive_threshold", &adaptive_threshold_tensor, "adaptive threshold kernel");
}