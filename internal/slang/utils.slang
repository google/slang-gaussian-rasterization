// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

float eps = 1e-7;

[Differentiable]
float3 read_xyz(uint32_t idx, DiffTensorView xyz)
{
    return float3(xyz[uint2(idx, 0)],
                  xyz[uint2(idx, 1)],
                  xyz[uint2(idx, 2)]);
}

[Differentiable]
float3 read_rgb(uint32_t idx, DiffTensorView rgb)
{
    return float3(rgb[uint2(idx, 0)],
                  rgb[uint2(idx, 1)],
                  rgb[uint2(idx, 2)]);
}

[Differentiable]
float4 read_inv_cov(uint32_t idx, DiffTensorView inv_cov)
{
    return float4(inv_cov[uint3(idx, 0, 0)],
                  inv_cov[uint3(idx, 1, 0)],
                  inv_cov[uint3(idx, 0, 1)],
                  inv_cov[uint3(idx, 1, 1)]);
}

[Differentiable]
float read_opacity(uint32_t idx, DiffTensorView opacity)
{
    return opacity[uint2(idx, 0)];
}

[Differentiable]
float ndc2pix(float v, int S)
{
	return ((v + 1.0) * S - 1.0) * 0.5;
}

[Differentiable]
float pix2ndc(float v, int S)
{
    return (2.0 * v + 1.0) / S - 1.0;
}

float clip(float val, float min_val, float max_val)
{
    return max(min_val, min(max_val, val));
}

float3 indexToRandomColor(uint32_t index) {
    // Simple hash function (improve for better distribution)
    float hashedValue = frac(sin(float(index)) * 43758.5453123);

    // Map hashed value to RGB components
    float3 color = float3(
        hashedValue,                  // Red
        frac(hashedValue * 23.5453), // Green
        frac(hashedValue * 177.635)  // Blue
    );

    return color;
}

[Differentiable]
//[BackwardDerivative(bwd_safe_sigmoid)]
float safe_sigmoid(float x) {
    if (x >= 0.0)
        return 1. / (1. + exp(-x));
    else
        return exp(x) / (1. + exp(x));
}

void bwd_safe_sigmoid(inout DifferentialPair<float> x, float dx) {
    float y = exp(-abs(x.p)); // Use abs(x) to avoid overflow
    float.Differential.dadd(x.d, (dx*y) / ((1 + y) * (1 + y)));
}

[Differentiable]
float sigmoid(float x, float height, float slope, float offset) {
    return height * safe_sigmoid(slope * (x - offset));
}

static const int NUM_SIGMOIDS = 8;

struct sigmoids : IDifferentiable
{
    float[NUM_SIGMOIDS] height;
    float[NUM_SIGMOIDS] slope;
    float[NUM_SIGMOIDS] offset;
};

[Differentiable]
sigmoids load_sigmoids(uint32_t2 pix_coord,
                       DiffTensorView sig_heights,
                       DiffTensorView sig_slopes,
                       DiffTensorView sig_offsets)
{
    sigmoids s;
    [ForceUnroll]
    for (uint32_t sig_idx = 0; sig_idx < NUM_SIGMOIDS; sig_idx++) {
        s.height[sig_idx] = sig_heights[uint3(pix_coord.y, pix_coord.x, sig_idx)];
        s.slope[sig_idx] = sig_slopes[uint3(pix_coord.y, pix_coord.x, sig_idx)];
        s.offset[sig_idx] = sig_offsets[uint3(pix_coord.y, pix_coord.x, sig_idx)];
    }
    return s;
}

[Differentiable]
float4 compute_sum_of_sigmoids(Gaussian g, sigmoids s, float4 gauss_rgba)
{
    float T = 0.0;
    [ForceUnroll]
    for (uint32_t sig_idx = 0; sig_idx < NUM_SIGMOIDS; sig_idx++) {
        T += sigmoid(g.xyz_vs.z, s.height[sig_idx], s.slope[sig_idx], s.offset[sig_idx]);
    }
    return (1 - T) * gauss_rgba;
}


struct Camera : IDifferentiable
{
    float4x4 world_view_transform;
    float4x4 proj_mat;
    float fovy;
    float fovx;
    int H;
    int W;
}

Camera load_camera(TensorView<float> world_view_transform_t, TensorView<float> proj_mat_t, no_diff float fovy, no_diff float fovx, uint H, uint W) {
    float4x4 world_view_transform = float4x4(world_view_transform_t[uint2(0, 0)], world_view_transform_t[uint2(0, 1)], world_view_transform_t[uint2(0, 2)], world_view_transform_t[uint2(0, 3)],
                                             world_view_transform_t[uint2(1, 0)], world_view_transform_t[uint2(1, 1)], world_view_transform_t[uint2(1, 2)], world_view_transform_t[uint2(1, 3)],
                                             world_view_transform_t[uint2(2, 0)], world_view_transform_t[uint2(2, 1)], world_view_transform_t[uint2(2, 2)], world_view_transform_t[uint2(2, 3)],
                                             world_view_transform_t[uint2(3, 0)], world_view_transform_t[uint2(3, 1)], world_view_transform_t[uint2(3, 2)], world_view_transform_t[uint2(3, 3)]);
    
    float4x4 proj_mat = float4x4(proj_mat_t[uint2(0, 0)], proj_mat_t[uint2(0, 1)], proj_mat_t[uint2(0, 2)], proj_mat_t[uint2(0, 3)],
                                 proj_mat_t[uint2(1, 0)], proj_mat_t[uint2(1, 1)], proj_mat_t[uint2(1, 2)], proj_mat_t[uint2(1, 3)],
                                 proj_mat_t[uint2(2, 0)], proj_mat_t[uint2(2, 1)], proj_mat_t[uint2(2, 2)], proj_mat_t[uint2(2, 3)],
                                 proj_mat_t[uint2(3, 0)], proj_mat_t[uint2(3, 1)], proj_mat_t[uint2(3, 2)], proj_mat_t[uint2(3, 3)]);

    return { world_view_transform, proj_mat, fovy, fovx, H, W};
}

struct Gaussian : IDifferentiable
{
    float3 xyz_vs;
    float3 rgb;
    float opacity;
    float2x2 inv_cov_vs;
};

[Differentiable]
float3 geom_transform_points(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz / (p_out.w + 1e-8);
}

[Differentiable]
float3 geom_transform_points2(float3 point, float4x4 transf_matrix)
{
    float4 p_out = mul(transf_matrix, float4(point, 1.0));
    return p_out.xyz;
}

[Differentiable]
float3 project_point(float3 point, Camera cam) {
    float3 proj_point = geom_transform_points(point, mul(cam.proj_mat, cam.world_view_transform));
    float3 view_point = geom_transform_points2(point, cam.world_view_transform);
    proj_point.z = view_point.z;
    return proj_point;
}

[Differentiable]
float3x3 compute_jacobian(float3 xyz_ws, Camera cam) {
    float tan_half_fovx = tan(cam.fovx / 2.0);
    float tan_half_fovy = tan(cam.fovy / 2.0);
    float h_x = cam.W / (2.0 * tan_half_fovx);
    float h_y = cam.H / (2.0 * tan_half_fovy);

    float3 t = geom_transform_points(xyz_ws, cam.world_view_transform);

    const float limx = 1.3f * tan_half_fovx;
    const float limy = 1.3f * tan_half_fovy;
    const float txtz = t.x / t.z;
    const float tytz = t.y / t.z;
    t.x = min(limx, max(-limx, txtz)) * t.z;
    t.y = min(limy, max(-limy, tytz)) * t.z;

    float3x3 J = float3x3(h_x / t.z, 0.0, -(h_x * t.x) / (t.z * t.z),
                          0.0, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
                          0.0, 0.0, 0.0);

    return J;
}

[Differentiable]
float2x2 covariance_3d_to_2d(Camera cam, float3 xyz_ws, float3x3 cov_ws) {
    float3x3 R = (float3x3)cam.world_view_transform;
    float3x3 J = compute_jacobian(xyz_ws, cam);
    float3x3 cov_vs = mul(J, mul(R, mul(cov_ws, mul(transpose(R), transpose(J)))));
    cov_vs[0][0] += 0.3;
    cov_vs[1][1] += 0.3;

    return float2x2(cov_vs);
}

[Differentiable]
Gaussian load_gaussian(uint32_t g_idx,
                       DiffTensorView xyz_vs,
                       DiffTensorView inv_cov_vs,
                       DiffTensorView opacity,
                       DiffTensorView rgb)
{
    float3 g_xyz_vs = float3(xyz_vs[uint2(g_idx, 0)],
                             xyz_vs[uint2(g_idx, 1)],
                             xyz_vs[uint2(g_idx, 2)]);
    float3 g_rgb = float3(rgb[uint2(g_idx, 0)],
                          rgb[uint2(g_idx, 1)],
                          rgb[uint2(g_idx, 2)]);

    float g_opacity = opacity[uint2(g_idx, 0)];
    float2x2 g_inv_cov = float2x2(inv_cov_vs[uint3(g_idx, 0, 0)], inv_cov_vs[uint3(g_idx, 0, 1)],
                                  inv_cov_vs[uint3(g_idx, 1, 0)], inv_cov_vs[uint3(g_idx, 1, 1)]);

    return { g_xyz_vs, g_rgb, g_opacity, g_inv_cov };
}

[Differentiable]
float4 evaluate_gaussian(Gaussian g, 
                         float2 pix_coord,
                         uint32_t H,
                         uint32_t W)
{
    float3 g_xyz = g.xyz_vs;
    float3 g_rgb = g.rgb;
    float g_opacity = g.opacity;
    float2x2 g_inv_cov_vs = g.inv_cov_vs;

    float2 d = { pix_coord.x - ndc2pix(g_xyz.x, W),
                 pix_coord.y - ndc2pix(g_xyz.y, H) };
    float power = -0.5f * (g_inv_cov_vs[0][0] * d.x * d.x +
                           g_inv_cov_vs[1][1] * d.y * d.y + (g_inv_cov_vs[0][1] + g_inv_cov_vs[1][0]) * d.x * d.y);
    float alpha = min(0.99f, g_opacity * exp(power));
    float3 premult_rgb = g_rgb * alpha;

    return float4(premult_rgb, alpha);
}

[Differentiable]
float4 normalize_pixel_state(float total_T, float4 pixel_state)
{
    if (pixel_state.a < 1e-4) {
        return float4(0.0, 0.0, 0.0, 0.0);
    }
    else {
      float norm_factor = (1.0 - total_T) / (pixel_state.a + 1e-7);
      return float4(pixel_state.rgb * norm_factor, pixel_state.a);
    }
}

[Differentiable]
float power_transform(float z)
{
    // float l = 0.2;
    // return 1.0 - (1.0 / (1.0 + l * z));
    return z - 4.0311;
}


[Differentiable]
float[5] compute_partial_moments(float4 rgba, Gaussian g)
{
    //float z_t = power_transform(g.xyz.z);
    float z_t = g.xyz_vs.z;
    
    return { rgba.a, // WHERE IS ln(1-a) from Sec 3.1 of Moment OIT?
             rgba.a * z_t,
             rgba.a * z_t * z_t,
             rgba.a * pow(z_t, 3),
             rgba.a * pow(z_t, 4) };
}

[Differentiable]
float[5] add_f5(float[5] this, float[5] other)
{
    return { this[0] + other[0],
             this[1] + other[1],
             this[2] + other[2],
             this[3] + other[3],
             this[4] + other[4] };
}

[Differentiable]
float[5] sub_f5(float[5] this, float[5] other)
{
    return { this[0] - other[0],
             this[1] - other[1],
             this[2] - other[2],
             this[3] - other[3],
             this[4] - other[4] };
}

[Differentiable]
float[5] update_moments(float[5] current, float[5] partial)
{
    return add_f5(current, partial);
}