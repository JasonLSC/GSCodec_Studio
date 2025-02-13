#include <torch/extension.h>

#include "gridencoder.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grid_encode_forward", &grid_encode_forward, "grid_encode_forward (CUDA)");
    m.def("grid_encode_backward", &grid_encode_backward, "grid_encode_backward (CUDA)");
    m.def("grid_encode_mix2D_forward", &grid_encode_mix2D_forward, "grid_encode_mix2D_forward (CUDA)");
    m.def("grid_encode_mix2D_backward", &grid_encode_mix2D_backward, "grid_encode_mix2D_backward (CUDA)");
    m.def("avg_2D_forward", &avg_2D_forward, "avg_2D_forward (CUDA)");
    m.def("avg_2D_backward", &avg_2D_backward, "avg_2D_backward (CUDA)");
    m.def("cnt_np_embed", &cnt_np_embed, "cnt_np_embed (CUDA)");
    m.def("cnt_np_embed_backward", &cnt_np_embed_backward, "cnt_np_embed_backward (CUDA)");
}