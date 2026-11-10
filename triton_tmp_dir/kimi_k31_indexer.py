# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 FP8 indexer weight scaling with checkpoint rounding order."""

from vllm.triton_utils import tl, triton


@triton.jit
def _scale_indexer_weights_kernel(
    values,
    scales,
    output,
    value_stride,
    head_count_scale,
    softmax_scale,
    NUM_HEADS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.arange(0, BLOCK)
    value = tl.load(values + row * value_stride + head, head < NUM_HEADS, 0)
    # The reference rounds head-count scaling back to projection dtype before
    # multiplying the FP8 query scale and indexer softmax scale in FP32.
    value = (value.to(tl.float32) * head_count_scale).to(values.dtype.element_ty)
    scale = tl.load(scales + row * NUM_HEADS + head, head < NUM_HEADS, 0)
    result = value.to(tl.float32) * scale * softmax_scale
    tl.store(output + row * NUM_HEADS + head, result, head < NUM_HEADS)


def scale_indexer_weights(values, scales, output, head_count_scale, softmax_scale):
    rows, heads = values.shape
    if rows == 0:
        return
    _scale_indexer_weights_kernel[(rows,)](
        values,
        scales,
        output,
        values.stride(0),
        head_count_scale,
        softmax_scale,
        NUM_HEADS=heads,
        BLOCK=triton.next_power_of_2(heads),
    )
