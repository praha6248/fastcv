#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

using u_char=unsigned char;
using coords=int;

#define max_block_size 31
#define cuda_block 16 //256 threads
__constant__ float kernel_gauss[1024]; //constant memory for gaussian kernel weights

enum AdaptiveMethod {
    ADAPTIVE_THRESH_MEAN_C = 0,
    ADAPTIVE_THRESH_GAUSSIAN_C = 1
};

enum ThresholdType {
    THRESH_BINARY = 0,
    THRESH_BINARY_INV = 1
};

// -----kernel 1-------
__global__ void ADAPTIVE_THRESH_MEAN_kernel(u_char* src, u_char* obraz_wyjsc, int width, int height, int image_step, double max_value, int threshold_type, int blockSize, double C) {
    __shared__ u_char shared_memory[cuda_block+max_block_size-1][cuda_block+max_block_size-1]; //shared memory for the block of the image plus neighboring pixels
    int r=blockSize/2; //block radius
    int tile_dim=cuda_block+2*r; //dimensions of the block of the image

    coords x=blockIdx.x*blockDim.x+threadIdx.x; 
    coords y=blockIdx.y*blockDim.y+threadIdx.y;

    coords local_x=threadIdx.x;
    coords local_y=threadIdx.y;

    //start positions for calculating mean 
    coords start_x=blockIdx.x*blockDim.x-r;
    coords start_y=blockIdx.y*blockDim.y-r;
    
    for (coords i=local_y; i<tile_dim; i+=blockDim.y){
        for (coords j=local_x; j<tile_dim; j+=blockDim.x){
            coords curr_x=start_x+j;
            coords curr_y=start_y+i; 

            if (curr_x<0) curr_x=0;
            if (curr_x>=width) curr_x=width-1;
            if (curr_y<0) curr_y = 0;
            if (curr_y>=height) curr_y=height-1;

            shared_memory[i][j] = src[curr_y*image_step+curr_x];
        }
    }

    __syncthreads();

    if (x<width && y<height) {
        float sum=0.0f;
        coords center_y=local_y+r;
        coords center_x=local_x+r;

        for (coords i=-r; i<=r; ++i){
            for (coords j=-r; j<=r; ++j) {
                sum+=shared_memory[center_y+i][center_x+j]; 
            }
        }

        float px_val=((float)C+shared_memory[local_y+r][local_x+r])*(float)(blockSize*blockSize);
        u_char wynik=0;
        
        //cv::THRESH_BINARY -> 0
        if (threshold_type==0){ 
            if (px_val>sum) {
                wynik=(u_char)max_value;
            } else{
                wynik=0;
            }
        } 
        else{ //cv::THRESH_BINARY_INV
            if (px_val>sum){
                wynik=0;
            } else{
                wynik=(u_char)max_value;
            }
        }
        //to global memory
        obraz_wyjsc[y*image_step+ x]=wynik;
    }
}

// -----kernel 2------
__global__ void gaussianAdaptiveThresholdKernel(u_char* src, u_char* dst, int width, int height, int image_step, double max_value, int threshold_type, int blockSize, int radius, double C){
    __shared__ u_char shared[(cuda_block + max_block_size - 1)][(cuda_block + max_block_size - 1)];
    coords x=blockIdx.x * blockDim.x + threadIdx.x;
    coords y=blockIdx.y * blockDim.y + threadIdx.y;

    int shared_s = cuda_block + blockSize - 1;

    for(int i = threadIdx.y; i < shared_s; i += cuda_block){
        for(int j = threadIdx.x; j < shared_s; j += cuda_block){
            if((x - threadIdx.x + j) < width && (y - threadIdx.y + i) < height){
                int img_x = x - threadIdx.x + j;
                int img_y = y - threadIdx.y + i;
                shared[i][j] = src[img_y * image_step + img_x];
            }
        }
    }

    __syncthreads();

    if(x >= width || y >= height){
        return;
    }

    float w_sum = 0.0f;

    for(int i = 0; i < blockSize; i++){
        for(int j = 0; j < blockSize; j++){
            u_char pix = shared[threadIdx.y + i][threadIdx.x + j];
            w_sum += pix * kernel_gauss[i * blockSize + j];
        }
    }
    float thresh = w_sum - (float)C;

    if(threshold_type == 0){
        dst[y * image_step + x] = (shared[threadIdx.y + radius][threadIdx.x + radius] > thresh) ? (u_char)max_value : 0;
    } else if(threshold_type == 1){
        dst[y * image_step + x] = (shared[threadIdx.y + radius][threadIdx.x + radius] > thresh) ? 0 : (u_char)max_value;
    }
}

// -----Gaussian kernel generation-----
struct Gaussian2D_w{
    int size;
    float sigma;
    int radius;
    //calculates the Gaussian weight for each point in 2D
    __device__ float operator()(int idx) const{
        //centers the row and column
        int x = (idx % size) - radius;
        int y = (idx / size) - radius;
        return expf(-(float)(x * x + y * y) / (2.0f * sigma * sigma));//Gaussian formula
    }
};

void generateGaussianKernel(int blockSize, float sigma, cudaStream_t stream){
    int total = blockSize * blockSize;
    int radius = blockSize / 2;
    //generating weights
    thrust::device_vector<float> d_kernel(total);
    thrust::transform(thrust::cuda::par.on(stream), thrust::make_counting_iterator(0), thrust::make_counting_iterator(total), d_kernel.begin(), Gaussian2D_w{blockSize, sigma, radius});
    //normalization
    float sum = thrust::reduce(thrust::cuda::par.on(stream), d_kernel.begin(), d_kernel.end(), 0.0f, thrust::plus<float>());
    thrust::transform(thrust::cuda::par.on(stream), d_kernel.begin(), d_kernel.end(), d_kernel.begin(), [sum] __device__ (float val) {
        return val / sum;
    });
    cudaMemcpyToSymbolAsync(kernel_gauss, thrust::raw_pointer_cast(d_kernel.data()), total * sizeof(float), 0, cudaMemcpyDeviceToDevice, stream);
}

void adaptiveThresholdCUDA(const u_char* src_data, u_char* dst_data, int width, int height, int step, double max_value, int method, int type, int blockSize, double delta){
    // -----walidation-----
    // standard 
    if (src_data == nullptr || dst_data == nullptr) {
        std::cerr << "Error" << std::endl;
        return;
    }
    
    if (blockSize % 2 == 0 || blockSize <= 1) {
        std::cerr << "Error" << std::endl;
        return;
    }
    
    // types walidation
    if (method != ADAPTIVE_THRESH_MEAN_C && method != ADAPTIVE_THRESH_GAUSSIAN_C) {
         std::cerr << "Error" << std::endl;
         return;
    }
    if (type != THRESH_BINARY && type != THRESH_BINARY_INV) {
        std::cerr << "Error" << std::endl;
        return;
    }

    if(max_value < 0){
        // dst = cv::Scalar(0);
        memset(dst_data, 0, step * height); 
        return;
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    //-----memory allocation (stream -> asynchronous)-----
    size_t dataSize = step * height;

    thrust::device_vector<u_char> d_src(dataSize); 
    thrust::device_vector<u_char> d_dst(dataSize);

    //src.data -> src_data
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_src.data()), src_data, dataSize, cudaMemcpyHostToDevice, stream);

    if(method == ADAPTIVE_THRESH_GAUSSIAN_C){
        generateGaussianKernel(blockSize, 0.3f * ((blockSize - 1) * 0.5f - 1) + 0.8f, stream); 
    }

    // -----grid and block-----
    dim3 threads_per_block(cuda_block, cuda_block);
    dim3 num_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

    // -----kernels-----
    {
        if(method == ADAPTIVE_THRESH_MEAN_C){
            ADAPTIVE_THRESH_MEAN_kernel<<<num_of_blocks, threads_per_block, 0, stream>>>(
                thrust::raw_pointer_cast(d_src.data()), 
                thrust::raw_pointer_cast(d_dst.data()),
                width,
                height,
                step,
                max_value,
                type,
                blockSize,
                delta
            );
        } else {
            gaussianAdaptiveThresholdKernel<<<num_of_blocks, threads_per_block, 0, 
                stream>>>(thrust::raw_pointer_cast(d_src.data()), thrust::raw_pointer_cast(d_dst.data()),
                    width, height, step, max_value, type, blockSize, blockSize/2, delta
                );
        }
    
        // -----getting result-----
        {
            cudaMemcpyAsync(dst_data, thrust::raw_pointer_cast(d_dst.data()), dataSize, cudaMemcpyDeviceToHost, stream);
        }
    
        cudaStreamSynchronize(stream);
    }
    cudaStreamDestroy(stream);
}

// Function to change input and output to torch tensors
torch::Tensor adaptive_threshold_tensor(torch::Tensor _src, double max_value, int method, int type, int blockSize, double delta){
    TORCH_CHECK(_src.device().type() == torch::kCPU, "Input must be a CPU tensor");
    TORCH_CHECK(_src.dtype() == torch::kByte, "Input must be uint8");
    
    const int height = _src.size(0);
    const int width = _src.size(1);
    const int step = _src.stride(0);

    auto _dst = torch::empty_like(_src);

    u_char* src_ptr = _src.data_ptr<u_char>();
    u_char* dst_ptr = _dst.data_ptr<u_char>();

    adaptiveThresholdCUDA(
        src_ptr,
        dst_ptr,
        width,
        height,
        step,
        max_value,
        method,
        type,
        blockSize,
        delta
    );

    return _dst;
}