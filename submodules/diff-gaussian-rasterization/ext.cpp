/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA); // ESTE SE USA EN TODOS LOS CASOS
  m.def("mark_visible", &markVisible); // ESTE ESTÁ DEFINIDO AUNQUE NO SE DONDE SE USA
  m.def("count_gaussians", &CountGaussiansCUDA);
  m.def("bw_score_gaussians", &BlendingWeightGaussiansCUDA);
  m.def("mw_score_gaussians", &MaxWeightGaussiansCUDA);
  m.def("topk_gaussians", &TopKGaussiansCUDA);
  m.def("topk_color_gaussians", &TopKColorGaussiansCUDA); // ESTE LO USO PARA SAFEGUARD
  m.def("topk_weight_gaussians", &TopKWeightGaussiansCUDA);
}