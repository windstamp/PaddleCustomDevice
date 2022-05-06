// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

#include "paddle/phi/core/tensor_meta.h"

namespace custom_kernel {

template <typename T, typename Context>
void FillConstantKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DenseTensorMeta::DataType dtype,
                phi::DenseTensor* out) {
  auto shape_vec = shape.GetData();
  out->ResizeAndAllocate(phi::make_ddim(shape_vec));
  dev_ctx.template Alloc<T>(out);

  aclrtStream stream = static_cast<aclrtStream>(dev_ctx.stream());
  T value = val.to<T>();

  if (dtype != phi::DenseTensorMeta::DataType::BOOL) {
    phi::DenseTensor tensor_value;
    tensor_value.Resize(phi::make_ddim({1}));
    FillNpuTensorWithConstant<T>(&tensor_value, dev_ctx, value);
    NpuOpRunner runner;
#if (CANN_VERSION_CODE >= 503003 && CANN_VERSION_CODE < 504001)
    runner.SetType("FillD")
        .AddInput(tensor_value)
        .AddOutput(*out)
        .AddAttrs({{"dims", shape_vec}})
        .Run(stream);
#else
    runner.SetType("Fill")
        .AddInput(dev_ctx, std::vector<int64_t>(shape_vec))
        .AddInput(tensor_value)
        .AddOutput(*out)
        .Run(stream);
#endif
  } else {
    auto op_func = [&shape_vec, &value](
        const std::vector<phi::DenseTensor>& inputs,
        const std::vector<phi::DenseTensor>& outputs,
        const NPUAttributeMap& attrs,
        const Context& dev_ctx) {
      phi::DenseTensor tensor_value;
      tensor_value.Resize(phi::make_ddim({1}));
      FillNpuTensorWithConstant<uint8_t>(
          &tensor_value, dev_ctx, static_cast<uint8_t>(value));

      NpuOpRunner runner;
      runner.SetType("Fill")
          .AddInput(dev_ctx, std::vector<int64_t>(shape_vec))
          .AddInput(tensor_value)
          .AddOutput(outputs[0])
          .Run(dev_ctx.stream());
    };
    NpuOpRunner::TypeAdapter({},
                             {*out},
                             {},
                             dev_ctx,
                             op_func,
                             {},
                             {phi::DenseTensorMeta::DataType::UINT8});
  }
}


}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(fill_constant,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::FillConstantKernel,
                          int8_t,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          bool) {}
