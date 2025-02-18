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
void FullKernel(const Context& dev_ctx,
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
#if (CANN_VERSION_CODE >= 503003)
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

template <typename T, typename Context>
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  using CommonType = typename std::common_type<
      float,
      typename std::conditional<std::is_same<T, phi::dtype::float16>::value,
                                float,
                                T>::type>::type;
  dev_ctx.template Alloc<T>(out);
  auto value = val.to<float>();

  auto common_type_value = static_cast<CommonType>(value);

  PADDLE_ENFORCE_EQ(
      (common_type_value >=
       static_cast<CommonType>(std::numeric_limits<T>::lowest())) &&
          (common_type_value <=
           static_cast<CommonType>(std::numeric_limits<T>::max())),
      true,
      phi::errors::InvalidArgument(
          "The filled value is out of range for target type, "
          "current kernel type is %s, the range should between %f "
          "and %f, but now value is %f.",
          typeid(T).name(),
          static_cast<CommonType>(std::numeric_limits<T>::lowest()),
          static_cast<CommonType>(std::numeric_limits<T>::max()),
          value));

  PADDLE_ENFORCE_EQ(std::isnan(value),
                    false,
                    phi::errors::InvalidArgument("The filled value is NaN."));

  phi::DenseTensor tensor_tmp;
  phi::DenseTensorMeta tensor_tmp_meta = {dtype, {1}};
  tensor_tmp.set_meta(tensor_tmp_meta);
  FillNpuTensorWithConstant<T>(&tensor_tmp, dev_ctx, static_cast<T>(value));

  auto stream = dev_ctx.stream();

  auto shape = out->dims();
  NpuOpRunner runner;
  runner.SetType("Fill")
      .AddInput(dev_ctx, phi::vectorize(shape))
      .AddInput(tensor_tmp)
      .AddOutput(*out)
      .Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(full,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::FullKernel,
                          int8_t,
                          int32_t,
                          int64_t,
                          float,
                          double,
                          bool) {}

PD_REGISTER_PLUGIN_KERNEL(full_like,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::FullLikeKernel,
                          float,
                          double,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}
