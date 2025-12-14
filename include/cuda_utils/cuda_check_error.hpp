
#ifndef CUDA_UTILS__CUDA_CHECK_ERROR_HPP_
#define CUDA_UTILS__CUDA_CHECK_ERROR_HPP_

#include <cuda_runtime_api.h>

#include <sstream>
#include <stdexcept>

namespace cuda_utils
{
template<typename F, typename N>
void cuda_check_error(const ::cudaError_t e, F&& f, N&& n)
{
    if (e != ::cudaSuccess)
    {
        std::stringstream s;
        s << ::cudaGetErrorName(e) << " (" << e << ")@" << f << "#L" << n << ": "
          << ::cudaGetErrorString(e);
        throw std::runtime_error {s.str()};
    }
}
}  // namespace cuda_utils

#define CHECK_CUDA_ERROR(e) (cuda_utils::cuda_check_error(e, __FILE__, __LINE__))

#endif  // CUDA_UTILS__CUDA_CHECK_ERROR_HPP_
