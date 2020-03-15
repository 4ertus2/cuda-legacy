#include <vector>
#include <iostream>

#define cudaErrCheck(code) if (code != cudaSuccess) throw CudaException{code, __FILE__, __LINE__}

struct CudaException
{
    cudaError_t code;
    const char * file;
    int line;
    
    const char * what() const noexcept { return cudaGetErrorString(code); }
};

__global__ void kernel(int * x, size_t size)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size)
        return;
    x[id] = id;
}

int main()
{
    try
    {
        size_t size = 1024;
        std::vector<int> x(size);

        int * dev_x = 0;
        cudaErrCheck(cudaMalloc((void **) &dev_x, size * sizeof(int)));
        cudaErrCheck(cudaMemcpy(dev_x, &x[0], size * sizeof(int), cudaMemcpyHostToDevice));

        size_t num_blocks = 16;
        size_t threads_per_block = 64;
        
        kernel<<<num_blocks, threads_per_block>>>(dev_x, size);
        cudaErrCheck(cudaPeekAtLastError());
        cudaErrCheck(cudaDeviceSynchronize());
        
        cudaErrCheck(cudaMemcpy(&x[0], dev_x, size * sizeof(int), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaFree(dev_x));
        
        for (size_t i = 0; i < size; ++i)
            if (i != x[i])
                std::cout << "not OK at position: " << i << std::endl;
        std::cout << "done" << std::endl;
    }
    catch (const CudaException & ex)
    {
        std::cerr << ex.file << ":" << ex.line << " " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
