

#ifndef CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_
#define CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_

#include <cuda_runtime_api.h>

#include <memory>

namespace cuda_utils
{
struct StreamDeleter
{
    void operator()(cudaStream_t* stream)
    {
        if (stream)
        {
            cudaStreamDestroy(*stream);
            delete stream;
        }
    }
};

using StreamUniquePtr = std::unique_ptr<cudaStream_t, StreamDeleter>;

inline StreamUniquePtr makeCudaStream(const uint32_t flags = cudaStreamDefault)
{
    StreamUniquePtr stream(new cudaStream_t, StreamDeleter());
    if (cudaStreamCreateWithFlags(stream.get(), flags) != cudaSuccess)
    {
        stream.reset(nullptr);
    }
    return stream;
}
}  // namespace cuda_utils

#endif  // CUDA_UTILS__STREAM_UNIQUE_PTR_HPP_
