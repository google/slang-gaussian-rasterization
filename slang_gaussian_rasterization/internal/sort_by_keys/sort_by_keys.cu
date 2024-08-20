#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <vector>

namespace extension_cpp {

  std::tuple<torch::Tensor, torch::Tensor>
  sort_by_keys(
    const at::Tensor keys,
    const at::Tensor values,
    const int highest_tile_id_msb)
  {
    TORCH_CHECK(keys.sizes() == values.sizes());
    TORCH_CHECK(keys.dtype() == torch::kLong);
    TORCH_CHECK(values.dtype() == torch::kInt32);
    TORCH_INTERNAL_ASSERT(keys.device().type() == at::DeviceType::CUDA);
    TORCH_INTERNAL_ASSERT(values.device().type() == at::DeviceType::CUDA);

    at::Tensor keys_sorted = torch::empty(keys.sizes(), keys.options());
    at::Tensor values_sorted = torch::empty(values.sizes(), values.options());

    at::Tensor keys_contig = keys.contiguous();
    at::Tensor values_contig = values.contiguous();
    at::Tensor keys_sorted_contig = keys_sorted.contiguous();
    at::Tensor values_sorted_contig = values_sorted.contiguous();

    const int64_t* keys_ptr = keys_contig.data_ptr<int64_t>();
    const int32_t* values_ptr = values_contig.data_ptr<int32_t>();
    int64_t* keys_sorted_ptr = keys_sorted_contig.data_ptr<int64_t>();
    int32_t* values_sorted_ptr = values_sorted_contig.data_ptr<int32_t>();

    void     *d_temp_storage = nullptr;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      keys_ptr, keys_sorted_ptr,
      values_ptr, values_sorted_ptr,
      keys.sizes()[0]);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
      d_temp_storage, temp_storage_bytes,
      keys_ptr, keys_sorted_ptr,
      values_ptr, values_sorted_ptr,
      keys.sizes()[0], 0, 32 + highest_tile_id_msb);

    cudaFree(d_temp_storage);

    return std::make_tuple(keys_sorted_contig, values_sorted_contig);
  }

  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sort_by_keys", &sort_by_keys);
  }

}
