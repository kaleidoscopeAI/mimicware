// spatial_optimizer.c - High performance spatial operations for diagram analysis

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // For AVX2 intrinsics

// Batch point-to-line distance calculation using AVX2
void batch_point_line_distance(
    float* points_x, float* points_y, // Array of point coordinates
    float* line_starts_x, float* line_starts_y, // Array of line start points
    float* line_ends_x, float* line_ends_y, // Array of line end points
    float* distances, // Output distances
    int num_points, int num_lines
) {
    // Process 8 points at once using AVX2
    for (int i = 0; i < num_lines; i++) {
        float x1 = line_starts_x[i];
        float y1 = line_starts_y[i];
        float x2 = line_ends_x[i];
        float y2 = line_ends_y[i];
        
        // Precompute line segments
        float line_length_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        __m256 v_x1 = _mm256_set1_ps(x1);
        __m256 v_y1 = _mm256_set1_ps(y1);
        __m256 v_x2 = _mm256_set1_ps(x2);
        __m256 v_y2 = _mm256_set1_ps(y2);
        __m256 v_line_length_sq = _mm256_set1_ps(line_length_sq);
        __m256 v_zero = _mm256_setzero_ps();
        __m256 v_one = _mm256_set1_ps(1.0f);
        
        // Process 8 points at a time using AVX2
        for (int j = 0; j < num_points; j += 8) {
            int remaining = num_points - j;
            int to_process = remaining < 8 ? remaining : 8;
            
            // Load 8 point coordinates (or fewer for the last batch)
            __m256 v_px, v_py;
            
            if (to_process == 8) {
                v_px = _mm256_loadu_ps(&points_x[j]);
                v_py = _mm256_loadu_ps(&points_y[j]);
            } else {
                // Handle the case where we have fewer than 8 points left
                float px_temp[8] = {0}, py_temp[8] = {0};
                for (int k = 0; k < to_process; k++) {
                    px_temp[k] = points_x[j + k];
                    py_temp[k] = points_y[j + k];
                }
                v_px = _mm256_loadu_ps(px_temp);
                v_py = _mm256_loadu_ps(py_temp);
            }
            
            // Calculate projection parameter t = ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / line_length_sq
            __m256 v_px_minus_x1 = _mm256_sub_ps(v_px, v_x1);
            __m256 v_py_minus_y1 = _mm256_sub_ps(v_py, v_y1);
            __m256 v_x2_minus_x1 = _mm256_sub_ps(v_x2, v_x1);
            __m256 v_y2_minus_y1 = _mm256_sub_ps(v_y2, v_y1);
            
            __m256 v_dot1 = _mm256_mul_ps(v_px_minus_x1, v_x2_minus_x1);
            __m256 v_dot2 = _mm256_mul_ps(v_py_minus_y1, v_y2_minus_y1);
            __m256 v_dot = _mm256_add_ps(v_dot1, v_dot2);
            
            __m256 v_t = _mm256_div_ps(v_dot, v_line_length_sq);
            
            // Clamp t to range [0, 1]
            v_t = _mm256_max_ps(v_zero, v_t);
            v_t = _mm256_min_ps(v_one, v_t);
            
            // Calculate the closest point on the line segment
            __m256 v_closest_x = _mm256_add_ps(v_x1, _mm256_mul_ps(v_t, v_x2_minus_x1));
            __m256 v_closest_y = _mm256_add_ps(v_y1, _mm256_mul_ps(v_t, v_y2_minus_y1));
            
            // Calculate distance to the closest point
            __m256 v_dx = _mm256_sub_ps(v_px, v_closest_x);
            __m256 v_dy = _mm256_sub_ps(v_py, v_closest_y);
            __m256 v_dist_sq = _mm256_add_ps(_mm256_mul_ps(v_dx, v_dx), _mm256_mul_ps(v_dy, v_dy));
            __m256 v_dist = _mm256_sqrt_ps(v_dist_sq);
            
            // Store distances
            float dist_temp[8];
            _mm256_storeu_ps(dist_temp, v_dist);
            
            for (int k = 0; k < to_process; k++) {
                distances[(i * num_points) + j + k] = dist_temp[k];
            }
        }
    }
}

// Batch point in polygon test using ray casting algorithm
void batch_point_in_polygon(
    float* points_x, float* points_y, // Array of test points
    float* polygon_x, float* polygon_y, // Polygon vertices
    int* results, // Output: 1 if point in polygon, 0 otherwise
    int num_points, int num_vertices
) {
    for (int i = 0; i < num_points; i++) {
        float x = points_x[i];
        float y = points_y[i];
        int inside = 0;
        
        // Ray casting algorithm implemented with SIMD support
        // Parts of this implementation use assembly directly for maximum performance
        
        __asm__ volatile(
            // Initialize counters and flags
            "xor %%eax, %%eax\n"         // inside flag = 0
            "mov %3, %%r8d\n"            // r8d = num_vertices
            "xorps %%xmm7, %%xmm7\n"     // zero register
            "movss %0, %%xmm0\n"         // xmm0 = x
            "movss %1, %%xmm1\n"         // xmm1 = y
            
            // Start the loop over polygon vertices
            "xor %%ecx, %%ecx\n"         // Initialize vertex index = 0
            "1:\n"                       // Start of loop
            
            // Load current vertex
            "mov %%ecx, %%edx\n"
            "shl $2, %%edx\n"            // edx = vertex_index * 4 (bytes per float)
            "movss (%4, %%rdx), %%xmm2\n" // xmm2 = polygon_x[vertex_index]
            "movss (%5, %%rdx), %%xmm3\n" // xmm3 = polygon_y[vertex_index]
            
            // Load next vertex (with wrap-around)
            "lea 1(%%ecx), %%edx\n"
            "cmp %%r8d, %%edx\n"
            "cmovae %%edi, %%edx\n"      // If next_index >= num_vertices, next_index = 0
            "shl $2, %%edx\n"            // edx = next_index * 4
            "movss (%4, %%rdx), %%xmm4\n" // xmm4 = polygon_x[next_index]
            "movss (%5, %%rdx), %%xmm5\n" // xmm5 = polygon_y[next_index]
            
            // Check if point is on line segment
            "movss %%xmm2, %%xmm6\n"     // Copy vertex coordinates for line equation
            "subss %%xmm4, %%xmm6\n"     // xmm6 = x1 - x2
            "movss %%xmm3, %%xmm7\n"
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y1 - y2
            
            // Ray casting test
            "ucomiss %%xmm1, %%xmm3\n"   // Compare y with y1
            "jb 2f\n"                    // If y < y1, jump to label 2
            "ucomiss %%xmm1, %%xmm5\n"   // Compare y with y2
            "jae 2f\n"                   // If y >= y2, jump to label 2
            
            // Calculate intersection of horizontal ray with edge
            "movss %%xmm3, %%xmm6\n"     // xmm6 = y1
            "subss %%xmm5, %%xmm6\n"     // xmm6 = y1 - y2
            "movss %%xmm1, %%xmm7\n"     // xmm7 = y
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y - y2
            "divss %%xmm6, %%xmm7\n"     // xmm7 = (y - y2) / (y1 - y2)
            "movss %%xmm2, %%xmm6\n"     // xmm6 = x1
            // spatial_optimizer.c - High performance spatial operations for diagram analysis

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // For AVX2 intrinsics

// Batch point-to-line distance calculation using AVX2
void batch_point_line_distance(
    float* points_x, float* points_y, // Array of point coordinates
    float* line_starts_x, float* line_starts_y, // Array of line start points
    float* line_ends_x, float* line_ends_y, // Array of line end points
    float* distances, // Output distances
    int num_points, int num_lines
) {
    // Process 8 points at once using AVX2
    for (int i = 0; i < num_lines; i++) {
        float x1 = line_starts_x[i];
        float y1 = line_starts_y[i];
        float x2 = line_ends_x[i];
        float y2 = line_ends_y[i];
        
        // Precompute line segments
        float line_length_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        __m256 v_x1 = _mm256_set1_ps(x1);
        __m256 v_y1 = _mm256_set1_ps(y1);
        __m256 v_x2 = _mm256_set1_ps(x2);
        __m256 v_y2 = _mm256_set1_ps(y2);
        __m256 v_line_length_sq = _mm256_set1_ps(line_length_sq);
        __m256 v_zero = _mm256_setzero_ps();
        __m256 v_one = _mm256_set1_ps(1.0f);
        
        // Process 8 points at a time using AVX2
        for (int j = 0; j < num_points; j += 8) {
            int remaining = num_points - j;
            int to_process = remaining < 8 ? remaining : 8;
            
            // Load 8 point coordinates (or fewer for the last batch)
            __m256 v_px, v_py;
            
            if (to_process == 8) {
                v_px = _mm256_loadu_ps(&points_x[j]);
                v_py = _mm256_loadu_ps(&points_y[j]);
            } else {
                // Handle the case where we have fewer than 8 points left
                float px_temp[8] = {0}, py_temp[8] = {0};
                for (int k = 0; k < to_process; k++) {
                    px_temp[k] = points_x[j + k];
                    py_temp[k] = points_y[j + k];
                }
                v_px = _mm256_loadu_ps(px_temp);
                v_py = _mm256_loadu_ps(py_temp);
            }
            
            // Calculate projection parameter t = ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / line_length_sq
            __m256 v_px_minus_x1 = _mm256_sub_ps(v_px, v_x1);
            __m256 v_py_minus_y1 = _mm256_sub_ps(v_py, v_y1);
            __m256 v_x2_minus_x1 = _mm256_sub_ps(v_x2, v_x1);
            __m256 v_y2_minus_y1 = _mm256_sub_ps(v_y2, v_y1);
            
            __m256 v_dot1 = _mm256_mul_ps(v_px_minus_x1, v_x2_minus_x1);
            __m256 v_dot2 = _mm256_mul_ps(v_py_minus_y1, v_y2_minus_y1);
            __m256 v_dot = _mm256_add_ps(v_dot1, v_dot2);
            
            __m256 v_t = _mm256_div_ps(v_dot, v_line_length_sq);
            
            // Clamp t to range [0, 1]
            v_t = _mm256_max_ps(v_zero, v_t);
            v_t = _mm256_min_ps(v_one, v_t);
            
            // Calculate the closest point on the line segment
            __m256 v_closest_x = _mm256_add_ps(v_x1, _mm256_mul_ps(v_t, v_x2_minus_x1));
            __m256 v_closest_y = _mm256_add_ps(v_y1, _mm256_mul_ps(v_t, v_y2_minus_y1));
            
            // Calculate distance to the closest point
            __m256 v_dx = _mm256_sub_ps(v_px, v_closest_x);
            __m256 v_dy = _mm256_sub_ps(v_py, v_closest_y);
            __m256 v_dist_sq = _mm256_add_ps(_mm256_mul_ps(v_dx, v_dx), _mm256_mul_ps(v_dy, v_dy));
            __m256 v_dist = _mm256_sqrt_ps(v_dist_sq);
            
            // Store distances
            float dist_temp[8];
            _mm256_storeu_ps(dist_temp, v_dist);
            
            for (int k = 0; k < to_process; k++) {
                distances[(i * num_points) + j + k] = dist_temp[k];
            }
        }
    }
}

// Batch point in polygon test using ray casting algorithm
void batch_point_in_polygon(
    float* points_x, float* points_y, // Array of test points
    float* polygon_x, float* polygon_y, // Polygon vertices
    int* results, // Output: 1 if point in polygon, 0 otherwise
    int num_points, int num_vertices
) {
    for (int i = 0; i < num_points; i++) {
        float x = points_x[i];
        float y = points_y[i];
        int inside = 0;
        
        // Ray casting algorithm implemented with SIMD support
        // Parts of this implementation use assembly directly for maximum performance
        
        __asm__ volatile(
            // Initialize counters and flags
            "xor %%eax, %%eax\n"         // inside flag = 0
            "mov %3, %%r8d\n"            // r8d = num_vertices
            "xorps %%xmm7, %%xmm7\n"     // zero register
            "movss %0, %%xmm0\n"         // xmm0 = x
            "movss %1, %%xmm1\n"         // xmm1 = y
            
            // Start the loop over polygon vertices
            "xor %%ecx, %%ecx\n"         // Initialize vertex index = 0
            "1:\n"                       // Start of loop
            
            // Load current vertex
            "mov %%ecx, %%edx\n"
            "shl $2, %%edx\n"            // edx = vertex_index * 4 (bytes per float)
            "movss (%4, %%rdx), %%xmm2\n" // xmm2 = polygon_x[vertex_index]
            "movss (%5, %%rdx), %%xmm3\n" // xmm3 = polygon_y[vertex_index]
            
            // Load next vertex (with wrap-around)
            "lea 1(%%ecx), %%edx\n"
            "cmp %%r8d, %%edx\n"
            "cmovae %%edi, %%edx\n"      // If next_index >= num_vertices, next_index = 0
            "shl $2, %%edx\n"            // edx = next_index * 4
            "movss (%4, %%rdx), %%xmm4\n" // xmm4 = polygon_x[next_index]
            "movss (%5, %%rdx), %%xmm5\n" // xmm5 = polygon_y[next_index]
            
            // Check if point is on line segment
            "movss %%xmm2, %%xmm6\n"     // Copy vertex coordinates for line equation
            "subss %%xmm4, %%xmm6\n"     // xmm6 = x1 - x2
            "movss %%xmm3, %%xmm7\n"
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y1 - y2
            
            // Ray casting test
            "ucomiss %%xmm1, %%xmm3\n"   // Compare y with y1
            "jb 2f\n"                    // If y < y1, jump to label 2
            "ucomiss %%xmm1, %%xmm5\n"   // Compare y with y2
            "jae 2f\n"                   // If y >= y2, jump to label 2
            
            // Calculate intersection of horizontal ray with edge
            "movss %%xmm3, %%xmm6\n"     // xmm6 = y1
            "subss %%xmm5, %%xmm6\n"     // xmm6 = y1 - y2
            "movss %%xmm1, %%xmm7\n"     // xmm7 = y
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y - y2
            "divss %%xmm6, %%xmm7\n"     // xmm7 = (y - y2) / (y1 - y2)
            "movss %%xmm2, %%xmm6\n"     // xmm6 = x1
            // spatial_optimizer.c - High performance spatial operations for diagram analysis

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // For AVX2 intrinsics

// Batch point-to-line distance calculation using AVX2
void batch_point_line_distance(
    float* points_x, float* points_y, // Array of point coordinates
    float* line_starts_x, float* line_starts_y, // Array of line start points
    float* line_ends_x, float* line_ends_y, // Array of line end points
    float* distances, // Output distances
    int num_points, int num_lines
) {
    // Process 8 points at once using AVX2
    for (int i = 0; i < num_lines; i++) {
        float x1 = line_starts_x[i];
        float y1 = line_starts_y[i];
        float x2 = line_ends_x[i];
        float y2 = line_ends_y[i];
        
        // Precompute line segments
        float line_length_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        __m256 v_x1 = _mm256_set1_ps(x1);
        __m256 v_y1 = _mm256_set1_ps(y1);
        __m256 v_x2 = _mm256_set1_ps(x2);
        __m256 v_y2 = _mm256_set1_ps(y2);
        __m256 v_line_length_sq = _mm256_set1_ps(line_length_sq);
        __m256 v_zero = _mm256_setzero_ps();
        __m256 v_one = _mm256_set1_ps(1.0f);
        
        // Process 8 points at a time using AVX2
        for (int j = 0; j < num_points; j += 8) {
            int remaining = num_points - j;
            int to_process = remaining < 8 ? remaining : 8;
            
            // Load 8 point coordinates (or fewer for the last batch)
            __m256 v_px, v_py;
            
            if (to_process == 8) {
                v_px = _mm256_loadu_ps(&points_x[j]);
                v_py = _mm256_loadu_ps(&points_y[j]);
            } else {
                // Handle the case where we have fewer than 8 points left
                float px_temp[8] = {0}, py_temp[8] = {0};
                for (int k = 0; k < to_process; k++) {
                    px_temp[k] = points_x[j + k];
                    py_temp[k] = points_y[j + k];
                }
                v_px = _mm256_loadu_ps(px_temp);
                v_py = _mm256_loadu_ps(py_temp);
            }
            
            // Calculate projection parameter t = ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / line_length_sq
            __m256 v_px_minus_x1 = _mm256_sub_ps(v_px, v_x1);
            __m256 v_py_minus_y1 = _mm256_sub_ps(v_py, v_y1);
            __m256 v_x2_minus_x1 = _mm256_sub_ps(v_x2, v_x1);
            __m256 v_y2_minus_y1 = _mm256_sub_ps(v_y2, v_y1);
            
            __m256 v_dot1 = _mm256_mul_ps(v_px_minus_x1, v_x2_minus_x1);
            __m256 v_dot2 = _mm256_mul_ps(v_py_minus_y1, v_y2_minus_y1);
            __m256 v_dot = _mm256_add_ps(v_dot1, v_dot2);
            
            __m256 v_t = _mm256_div_ps(v_dot, v_line_length_sq);
            
            // Clamp t to range [0, 1]
            v_t = _mm256_max_ps(v_zero, v_t);
            v_t = _mm256_min_ps(v_one, v_t);
            
            // Calculate the closest point on the line segment
            __m256 v_closest_x = _mm256_add_ps(v_x1, _mm256_mul_ps(v_t, v_x2_minus_x1));
            __m256 v_closest_y = _mm256_add_ps(v_y1, _mm256_mul_ps(v_t, v_y2_minus_y1));
            
            // Calculate distance to the closest point
            __m256 v_dx = _mm256_sub_ps(v_px, v_closest_x);
            __m256 v_dy = _mm256_sub_ps(v_py, v_closest_y);
            __m256 v_dist_sq = _mm256_add_ps(_mm256_mul_ps(v_dx, v_dx), _mm256_mul_ps(v_dy, v_dy));
            __m256 v_dist = _mm256_sqrt_ps(v_dist_sq);
            
            // Store distances
            float dist_temp[8];
            _mm256_storeu_ps(dist_temp, v_dist);
            
            for (int k = 0; k < to_process; k++) {
                distances[(i * num_points) + j + k] = dist_temp[k];
            }
        }
    }
}

// Batch point in polygon test using ray casting algorithm
void batch_point_in_polygon(
    float* points_x, float* points_y, // Array of test points
    float* polygon_x, float* polygon_y, // Polygon vertices
    int* results, // Output: 1 if point in polygon, 0 otherwise
    int num_points, int num_vertices
) {
    for (int i = 0; i < num_points; i++) {
        float x = points_x[i];
        float y = points_y[i];
        int inside = 0;
        
        // Ray casting algorithm implemented with SIMD support
        // Parts of this implementation use assembly directly for maximum performance
        
        __asm__ volatile(
            // Initialize counters and flags
            "xor %%eax, %%eax\n"         // inside flag = 0
            "mov %3, %%r8d\n"            // r8d = num_vertices
            "xorps %%xmm7, %%xmm7\n"     // zero register
            "movss %0, %%xmm0\n"         // xmm0 = x
            "movss %1, %%xmm1\n"         // xmm1 = y
            
            // Start the loop over polygon vertices
            "xor %%ecx, %%ecx\n"         // Initialize vertex index = 0
            "1:\n"                       // Start of loop
            
            // Load current vertex
            "mov %%ecx, %%edx\n"
            "shl $2, %%edx\n"            // edx = vertex_index * 4 (bytes per float)
            "movss (%4, %%rdx), %%xmm2\n" // xmm2 = polygon_x[vertex_index]
            "movss (%5, %%rdx), %%xmm3\n" // xmm3 = polygon_y[vertex_index]
            
            // Load next vertex (with wrap-around)
            "lea 1(%%ecx), %%edx\n"
            "cmp %%r8d, %%edx\n"
            "cmovae %%edi, %%edx\n"      // If next_index >= num_vertices, next_index = 0
            "shl $2, %%edx\n"            // edx = next_index * 4
            "movss (%4, %%rdx), %%xmm4\n" // xmm4 = polygon_x[next_index]
            "movss (%5, %%rdx), %%xmm5\n" // xmm5 = polygon_y[next_index]
            
            // Check if point is on line segment
            "movss %%xmm2, %%xmm6\n"     // Copy vertex coordinates for line equation
            "subss %%xmm4, %%xmm6\n"     // xmm6 = x1 - x2
            "movss %%xmm3, %%xmm7\n"
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y1 - y2
            
            // Ray casting test
            "ucomiss %%xmm1, %%xmm3\n"   // Compare y with y1
            "jb 2f\n"                    // If y < y1, jump to label 2
            "ucomiss %%xmm1, %%xmm5\n"   // Compare y with y2
            "jae 2f\n"                   // If y >= y2, jump to label 2
            
            // Calculate intersection of horizontal ray with edge
            "movss %%xmm3, %%xmm6\n"     // xmm6 = y1
            "subss %%xmm5, %%xmm6\n"     // xmm6 = y1 - y2
            "movss %%xmm1, %%xmm7\n"     // xmm7 = y
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y - y2
            "divss %%xmm6, %%xmm7\n"     // xmm7 = (y - y2) / (y1 - y2)
            "movss %%xmm2, %%xmm6\n"     // xmm6 = x1
            // spatial_optimizer.c - High performance spatial operations for diagram analysis

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // For AVX2 intrinsics

// Batch point-to-line distance calculation using AVX2
void batch_point_line_distance(
    float* points_x, float* points_y, // Array of point coordinates
    float* line_starts_x, float* line_starts_y, // Array of line start points
    float* line_ends_x, float* line_ends_y, // Array of line end points
    float* distances, // Output distances
    int num_points, int num_lines
) {
    // Process 8 points at once using AVX2
    for (int i = 0; i < num_lines; i++) {
        float x1 = line_starts_x[i];
        float y1 = line_starts_y[i];
        float x2 = line_ends_x[i];
        float y2 = line_ends_y[i];
        
        // Precompute line segments
        float line_length_sq = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1);
        __m256 v_x1 = _mm256_set1_ps(x1);
        __m256 v_y1 = _mm256_set1_ps(y1);
        __m256 v_x2 = _mm256_set1_ps(x2);
        __m256 v_y2 = _mm256_set1_ps(y2);
        __m256 v_line_length_sq = _mm256_set1_ps(line_length_sq);
        __m256 v_zero = _mm256_setzero_ps();
        __m256 v_one = _mm256_set1_ps(1.0f);
        
        // Process 8 points at a time using AVX2
        for (int j = 0; j < num_points; j += 8) {
            int remaining = num_points - j;
            int to_process = remaining < 8 ? remaining : 8;
            
            // Load 8 point coordinates (or fewer for the last batch)
            __m256 v_px, v_py;
            
            if (to_process == 8) {
                v_px = _mm256_loadu_ps(&points_x[j]);
                v_py = _mm256_loadu_ps(&points_y[j]);
            } else {
                // Handle the case where we have fewer than 8 points left
                float px_temp[8] = {0}, py_temp[8] = {0};
                for (int k = 0; k < to_process; k++) {
                    px_temp[k] = points_x[j + k];
                    py_temp[k] = points_y[j + k];
                }
                v_px = _mm256_loadu_ps(px_temp);
                v_py = _mm256_loadu_ps(py_temp);
            }
            
            // Calculate projection parameter t = ((px-x1)*(x2-x1) + (py-y1)*(y2-y1)) / line_length_sq
            __m256 v_px_minus_x1 = _mm256_sub_ps(v_px, v_x1);
            __m256 v_py_minus_y1 = _mm256_sub_ps(v_py, v_y1);
            __m256 v_x2_minus_x1 = _mm256_sub_ps(v_x2, v_x1);
            __m256 v_y2_minus_y1 = _mm256_sub_ps(v_y2, v_y1);
            
            __m256 v_dot1 = _mm256_mul_ps(v_px_minus_x1, v_x2_minus_x1);
            __m256 v_dot2 = _mm256_mul_ps(v_py_minus_y1, v_y2_minus_y1);
            __m256 v_dot = _mm256_add_ps(v_dot1, v_dot2);
            
            __m256 v_t = _mm256_div_ps(v_dot, v_line_length_sq);
            
            // Clamp t to range [0, 1]
            v_t = _mm256_max_ps(v_zero, v_t);
            v_t = _mm256_min_ps(v_one, v_t);
            
            // Calculate the closest point on the line segment
            __m256 v_closest_x = _mm256_add_ps(v_x1, _mm256_mul_ps(v_t, v_x2_minus_x1));
            __m256 v_closest_y = _mm256_add_ps(v_y1, _mm256_mul_ps(v_t, v_y2_minus_y1));
            
            // Calculate distance to the closest point
            __m256 v_dx = _mm256_sub_ps(v_px, v_closest_x);
            __m256 v_dy = _mm256_sub_ps(v_py, v_closest_y);
            __m256 v_dist_sq = _mm256_add_ps(_mm256_mul_ps(v_dx, v_dx), _mm256_mul_ps(v_dy, v_dy));
            __m256 v_dist = _mm256_sqrt_ps(v_dist_sq);
            
            // Store distances
            float dist_temp[8];
            _mm256_storeu_ps(dist_temp, v_dist);
            
            for (int k = 0; k < to_process; k++) {
                distances[(i * num_points) + j + k] = dist_temp[k];
            }
        }
    }
}

// Batch point in polygon test using ray casting algorithm
void batch_point_in_polygon(
    float* points_x, float* points_y, // Array of test points
    float* polygon_x, float* polygon_y, // Polygon vertices
    int* results, // Output: 1 if point in polygon, 0 otherwise
    int num_points, int num_vertices
) {
    for (int i = 0; i < num_points; i++) {
        float x = points_x[i];
        float y = points_y[i];
        int inside = 0;
        
        // Ray casting algorithm implemented with SIMD support
        // Parts of this implementation use assembly directly for maximum performance
        
        __asm__ volatile(
            // Initialize counters and flags
            "xor %%eax, %%eax\n"         // inside flag = 0
            "mov %3, %%r8d\n"            // r8d = num_vertices
            "xorps %%xmm7, %%xmm7\n"     // zero register
            "movss %0, %%xmm0\n"         // xmm0 = x
            "movss %1, %%xmm1\n"         // xmm1 = y
            
            // Start the loop over polygon vertices
            "xor %%ecx, %%ecx\n"         // Initialize vertex index = 0
            "1:\n"                       // Start of loop
            
            // Load current vertex
            "mov %%ecx, %%edx\n"
            "shl $2, %%edx\n"            // edx = vertex_index * 4 (bytes per float)
            "movss (%4, %%rdx), %%xmm2\n" // xmm2 = polygon_x[vertex_index]
            "movss (%5, %%rdx), %%xmm3\n" // xmm3 = polygon_y[vertex_index]
            
            // Load next vertex (with wrap-around)
            "lea 1(%%ecx), %%edx\n"
            "cmp %%r8d, %%edx\n"
            "cmovae %%edi, %%edx\n"      // If next_index >= num_vertices, next_index = 0
            "shl $2, %%edx\n"            // edx = next_index * 4
            "movss (%4, %%rdx), %%xmm4\n" // xmm4 = polygon_x[next_index]
            "movss (%5, %%rdx), %%xmm5\n" // xmm5 = polygon_y[next_index]
            
            // Check if point is on line segment
            "movss %%xmm2, %%xmm6\n"     // Copy vertex coordinates for line equation
            "subss %%xmm4, %%xmm6\n"     // xmm6 = x1 - x2
            "movss %%xmm3, %%xmm7\n"
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y1 - y2
            
            // Ray casting test
            "ucomiss %%xmm1, %%xmm3\n"   // Compare y with y1
            "jb 2f\n"                    // If y < y1, jump to label 2
            "ucomiss %%xmm1, %%xmm5\n"   // Compare y with y2
            "jae 2f\n"                   // If y >= y2, jump to label 2
            
            // Calculate intersection of horizontal ray with edge
            "movss %%xmm3, %%xmm6\n"     // xmm6 = y1
            "subss %%xmm5, %%xmm6\n"     // xmm6 = y1 - y2
            "movss %%xmm1, %%xmm7\n"     // xmm7 = y
            "subss %%xmm5, %%xmm7\n"     // xmm7 = y - y2
            "divss %%xmm6, %%xmm7\n"     // xmm7 = (y - y2) / (y1 - y2)
            "movss %%xmm2, %%xmm6\n"     // xmm6 = x1
            "subss %%xmm4, %%xmm6\n"     // xmm6 = x1 - x2
            "mulss %%xmm7, %%xmm6\n"     // xmm6 = (x1 - x2) * (y - y2) / (y1 - y2)
            "addss %%xmm4, %%xmm6\n"     // xmm6 = x2 + (x1 - x2) * (y - y2) / (y1 - y2) = intersection x
            
            // If intersection is to the right of our point, increment inside flag
            "ucomiss %%xmm0, %%xmm6\n"   // Compare x with intersection_x
            "jbe 2f\n"                   // If x >= intersection_x, jump to label 2
            "xor $1, %%eax\n"            // Toggle inside flag
            
            "2:\n"                       // Jump target for condition checks
            "inc %%ecx\n"                // Increment vertex index
            "cmp %%r8d, %%ecx\n"         // Compare with num_vertices
            "jb 1b\n"                    // If vertex_index < num_vertices, loop back
            
            // Store result in output array
            "mov %%eax, %2\n"            // Store inside flag in result variable
            
            : "=m" (results[i])          // Output
            : "m" (x), "m" (y), "m" (num_vertices), 
              "r" (polygon_x), "r" (polygon_y)
            : "eax", "ecx", "edx", "edi", "r8", "xmm0", "xmm1", "xmm2", 
              "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "memory"
        );
    }
}

// Optimized R-tree-like spatial index search
void rtree_nearest_neighbors(
    float* query_points_x, float* query_points_y, // Query points
    float* node_centers_x, float* node_centers_y, // Node centers (centroids of bounding boxes)
    float* node_extents_x, float* node_extents_y, // Half-extents of bounding boxes
    int* results_indices, float* results_distances, // Output: indices and distances
    int num_queries, int num_nodes, int k_neighbors
) {
    #pragma omp parallel for
    for (int i = 0; i < num_queries; i++) {
        float px = query_points_x[i];
        float py = query_points_y[i];
        
        // Priority queue structure (fixed-size k neighbors)
        int* neighbor_indices = (int*)malloc(k_neighbors * sizeof(int));
        float* neighbor_distances = (float*)malloc(k_neighbors * sizeof(float));
        
        // Initialize with invalid values
        for (int j = 0; j < k_neighbors; j++) {
            neighbor_indices[j] = -1;
            neighbor_distances[j] = INFINITY;
        }
        
        // Process in batches of 8 nodes using AVX2
        int batches = (num_nodes + 7) / 8;
        
        for (int batch = 0; batch < batches; batch++) {
            int batch_start = batch * 8;
            int batch_end = batch_start + 8;
            if (batch_end > num_nodes) batch_end = num_nodes;
            
            __m256 vx = _mm256_set1_ps(px);
            __m256 vy = _mm256_set1_ps(py);
            
            // Load 8 node centers
            float centers_x[8] = {0}, centers_y[8] = {0};
            float extents_x[8] = {0}, extents_y[8] = {0};
            
            for (int j = batch_start, idx = 0; j < batch_end; j++, idx++) {
                centers_x[idx] = node_centers_x[j];
                centers_y[idx] = node_centers_y[j];
                extents_x[idx] = node_extents_x[j];
                extents_y[idx] = node_extents_y[j];
            }
            
            __m256 vnode_x = _mm256_loadu_ps(centers_x);
            __m256 vnode_y = _mm256_loadu_ps(centers_y);
            __m256 vextent_x = _mm256_loadu_ps(extents_x);
            __m256 vextent_y = _mm256_loadu_ps(extents_y);
            
            // Calculate distances using AVX2
            __m256 vdx = _mm256_sub_ps(vx, vnode_x);
            __m256 vdy = _mm256_sub_ps(vy, vnode_y);
            
            // Get absolute values
            __m256 vabs_dx = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vdx);
            __m256 vabs_dy = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), vdy);
            
            // Compute max(0, abs(dx) - extent_x)
            __m256 vbox_dx = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(vabs_dx, vextent_x));
            __m256 vbox_dy = _mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(vabs_dy, vextent_y));
            
            // Square and add
            __m256 vsqr_dx = _mm256_mul_ps(vbox_dx, vbox_dx);
            __m256 vsqr_dy = _mm256_mul_ps(vbox_dy, vbox_dy);
            __m256 vsqr_dist = _mm256_add_ps(vsqr_dx, vsqr_dy);
            
            // Get square root for actual distances
            __m256 vdist = _mm256_sqrt_ps(vsqr_dist);
            
            // Store distances
            float distances[8];
            _mm256_storeu_ps(distances, vdist);
            
            // Update nearest neighbors
            for (int j = batch_start, idx = 0; j < batch_end; j++, idx++) {
                float dist = distances[idx];
                
                // Check if this node is closer than our current k-nearest neighbors
                for (int nn = 0; nn < k_neighbors; nn++) {
                    if (dist < neighbor_distances[nn]) {
                        // Shift all further items
                        for (int s = k_neighbors - 1; s > nn; s--) {
                            neighbor_indices[s] = neighbor_indices[s-1];
                            neighbor_distances[s] = neighbor_distances[s-1];
                        }
                        // Insert at the right position
                        neighbor_indices[nn] = j;
                        neighbor_distances[nn] = dist;
                        break;
                    }
                }
            }
        }
        
        // Copy results to output arrays
        for (int j = 0; j < k_neighbors; j++) {
            results_indices[i * k_neighbors + j] = neighbor_indices[j];
            results_distances[i * k_neighbors + j] = neighbor_distances[j];
        }
        
        free(neighbor_indices);
        free(neighbor_distances);
    }
}

// Optimized convex hull calculation using Graham scan
void compute_convex_hull(
    float* points_x, float* points_y, // Input points
    int* hull_indices, int* hull_size, // Output: indices of hull points and hull size
    int num_points
) {
    if (num_points < 3) {
        // Edge case: all points are in the hull
        *hull_size = num_points;
        for (int i = 0; i < num_points; i++) {
            hull_indices[i] = i;
        }
        return;
    }
    
    // Find point with lowest y-coordinate (and leftmost if tied)
    int anchor_idx = 0;
    float anchor_x = points_x[0];
    float anchor_y = points_y[0];
    
    for (int i = 1; i < num_points; i++) {
        if (points_y[i] < anchor_y || (points_y[i] == anchor_y && points_x[i] < anchor_x)) {
            anchor_idx = i;
            anchor_x = points_x[i];
            anchor_y = points_y[i];
        }
    }
    
    // Allocate memory for sorting
    typedef struct {
        int idx;
        float angle;
        float distance;
    } PointAngle;
    
    PointAngle* sorted_points = (PointAngle*)malloc((num_points - 1) * sizeof(PointAngle));
    
    // Compute angles relative to anchor point
    int sort_idx = 0;
    for (int i = 0; i < num_points; i++) {
        if (i == anchor_idx) continue;
        
        float dx = points_x[i] - anchor_x;
        float dy = points_y[i] - anchor_y;
        float angle = atan2f(dy, dx);
        float distance = sqrtf(dx*dx + dy*dy);
        
        sorted_points[sort_idx].idx = i;
        sorted_points[sort_idx].angle = angle;
        sorted_points[sort_idx].distance = distance;
        sort_idx++;
    }
    
    // Sort points by angle
    // Using insertion sort for simplicity, but quicksort would be better for larger point sets
    for (int i = 1; i < num_points - 1; i++) {
        PointAngle temp = sorted_points[i];
        int j = i - 1;
        
        while (j >= 0 && (sorted_points[j].angle > temp.angle || 
                          (sorted_points[j].angle == temp.angle && sorted_points[j].distance < temp.distance))) {
            sorted_points[j + 1] = sorted_points[j];
            j--;
        }
        
        sorted_points[j + 1] = temp;
    }
    
    // Graham scan algorithm
    hull_indices[0] = anchor_idx;
    hull_indices[1] = sorted_points[0].idx;
    
    int top = 1;
    
    for (int i = 1; i < num_points - 1; i++) {
        while (top >= 1 && !is_ccw(
            points_x[hull_indices[top-1]], points_y[hull_indices[top-1]],
            points_x[hull_indices[top]], points_y[hull_indices[top]],
            points_x[sorted_points[i].idx], points_y[sorted_points[i].idx])) {
            top--;
        }
        
        top++;
        hull_indices[top] = sorted_points[i].idx;
    }
    
    *hull_size = top + 1;
    free(sorted_points);
}

// Helper function: Check if three points make a counter-clockwise turn
int is_ccw(float x1, float y1, float x2, float y2, float x3, float y3) {
    return ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) > 0;
}

// Specialized optimization for UML class diagram relationship detection
void detect_uml_relationships(
    float* boxes_x, float* boxes_y, float* boxes_width, float* boxes_height, // Class boxes
    float* arrows_x1, float* arrows_y1, float* arrows_x2, float* arrows_y2,  // Arrows
    int* arrow_styles, // 0=solid, 1=dashed, 2=dotted
    int* arrow_heads,  // 0=none, 1=normal, 2=triangle, 3=diamond, 4=diamond_filled
    int* relationship_types, // Output: 0=association, 1=inheritance, 2=composition, etc.
    int* source_indices, int* target_indices, // Output: detected endpoints
    int num_boxes, int num_arrows
) {
    // Define relationship type constants
    const int REL_ASSOCIATION = 0;
    const int REL_INHERITANCE = 1;
    const int REL_COMPOSITION = 2;
    const int REL_AGGREGATION = 3;
    const int REL_DEPENDENCY = 4;
    const int REL_REALIZATION = 5;
    
    // Process each arrow to determine relationship type and endpoints
    for (int i = 0; i < num_arrows; i++) {
        float x1 = arrows_x1[i];
        float y1 = arrows_y1[i];
        float x2 = arrows_x2[i];
        float y2 = arrows_y2[i];
        int style = arrow_styles[i];
        int head = arrow_heads[i];
        
        // Determine relationship type based on arrow style and head
        int rel_type = REL_ASSOCIATION; // Default
        
        if (style == 1) { // Dashed
            rel_type = REL_DEPENDENCY;
            if (head == 2) { // Triangle
                rel_type = REL_REALIZATION;
            }
        } else { // Solid
            if (head == 2) { // Triangle
                rel_type = REL_INHERITANCE;
            } else if (head == 3) { // Diamond
                rel_type = REL_AGGREGATION;
            } else if (head == 4) { // Filled diamond
                rel_type = REL_COMPOSITION;
            }
        }
        
        relationship_types[i] = rel_type;
        
        // Find source and target boxes
        int source_idx = -1;
        int target_idx = -1;
        float min_source_dist = INFINITY;
        float min_target_dist = INFINITY;
        
        for (int j = 0; j < num_boxes; j++) {
            float box_x = boxes_x[j];
            float box_y = boxes_y[j];
            float box_w = boxes_width[j];
            float box_h = boxes_height[j];
            
            // Check if arrow endpoints are near this box
            if (point_near_box(x1, y1, box_x, box_y, box_w, box_h, 10.0f)) {
                float dist = point_to_box_distance(x1, y1, box_x, box_y, box_w, box_h);
                if (dist < min_source_dist) {
                    min_source_dist = dist;
                    source_idx = j;
                }
            }
            
            if (point_near_box(x2, y2, box_x, box_y, box_w, box_h, 10.0f)) {
                float dist = point_to_box_distance(x2, y2, box_x, box_y, box_w, box_h);
                if (dist < min_target_dist) {
                    min_target_dist = dist;
                    target_idx = j;
                }
            }
        }
        
        source_indices[i] = source_idx;
        target_indices[i] = target_idx;
    }
}

// Helper: Check if point is near a box
int point_near_box(
    float px, float py, float box_x, float box_y, float box_w, float box_h, float threshold
) {
    // Expand box by threshold
    float min_x = box_x - box_w/2 - threshold;
    float max_x = box_x + box_w/2 + threshold;
    float min_y = box_y - box_h/2 - threshold;
    float max_y = box_y + box_h/2 + threshold;
    
    return (px >= min_x && px <= max_x && py >= min_y && py <= max_y);
}

// Helper: Calculate distance from point to box
float point_to_box_distance(
    float px, float py, float box_x, float box_y, float box_w, float box_h
) {
    float min_x = box_x - box_w/2;
    float max_x = box_x + box_w/2;
    float min_y = box_y - box_h/2;
    float max_y = box_y + box_h/2;
    
    // Check if point is inside box
    if (px >= min_x && px <= max_x && py >= min_y && py <= max_y) {
        return 0.0f;
    }
    
    // Calculate distance to closest edge
    float dx = fmaxf(min_x - px, 0.0f);
    dx = fmaxf(dx, px - max_x);
    
    float dy = fmaxf(min_y - py, 0.0f);
    dy = fmaxf(dy, py - max_y);
    
    return sqrtf(dx*dx + dy*dy);
}

// Python API binding function
void init_spatial_optimizer(void) {
    // This function would set up Python bindings using e.g. pybind11 or ctypes
    // For the C functions above, allowing them to be called from Python
}
class GraphTheoreticDiagramAnalyzer:
    """Advanced graph theoretical analyzer for diagram structures using spectral techniques"""
    
    def __init__(self):
        self.graph = None
        self.eigenvector_centralities = None
        self.components = None
        self.spectral_clusters = None
    
    def analyze(self, nodes, connections):
        """Perform comprehensive graph-theoretic analysis on diagram"""
        import networkx as nx
        import numpy as np
        from sklearn.cluster import SpectralClustering
        
        # Create graph representation
        self.graph = nx.DiGraph()
        
        # Add nodes with attributes
        for node in nodes:
            self.graph.add_node(node['id'], **node)
        
        # Add edges with attributes
        for conn in connections:
            self.graph.add_edge(conn['source'], conn['target'], **conn)
        
        # Compute key metrics
        self._compute_centrality_measures()
        self._identify_communities()
        self._analyze_flow_patterns()
        self._detect_hierarchies()
        self._perform_spectral_analysis()
        
        # Return comprehensive analysis
        return self._compile_results()
    
    def _compute_centrality_measures(self):
        """Calculate various centrality measures"""
        import networkx as nx
        
        # Eigenvector centrality
        try:
            self.eigenvector_centralities = nx.eigenvector_centrality_numpy(self.graph)
        except:
            self.eigenvector_centralities = {}  # Fallback if calculation fails
            
        # Betweenness centrality (nodes that control information flow)
        try:
            self.betweenness = nx.betweenness_centrality(self.graph)
        except:
            self.betweenness = {}
            
        # PageRank (for analyzing influence)
        try:
            self.pageranks = nx.pagerank(self.graph)
        except:
            self.pageranks = {}
    
    def _identify_communities(self):
        """Identify communities/modules in the graph"""
        import networkx as nx
        import numpy as np
        from sklearn.cluster import SpectralClustering
        
        # Get undirected version for community detection
        undirected = self.graph.to_undirected()
        
        # Connected components
        self.components = list(nx.connected_components(undirected))
        
        # Try community detection if graph is large enough
        if len(self.graph) > 3:
            try:
                # Prepare adjacency matrix
                adj_matrix = nx.to_numpy_array(undirected)
                
                # Determine optimal number of clusters (simplified)
                n_clusters = min(max(2, len(self.graph) // 5), 8)
                
                # Spectral clustering
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    assign_labels='discretize',
                    random_state=42
                ).fit(adj_matrix)
                
                # Organize nodes by cluster
                self.spectral_clusters = [[] for _ in range(n_clusters)]
                for i, label in enumerate(clustering.labels_):
                    node_id = list(self.graph.nodes())[i]
                    self.spectral_clusters[label].append(node_id)
            except:
                # Fallback to simple connected components
                self.spectral_clusters = self.components
    
    def _analyze_flow_patterns(self):
        """Analyze flow patterns in directed graphs"""
        import networkx as nx
        
        # Sources (nodes with only outgoing edges)
        self.sources = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0 and self.graph.out_degree(n) > 0]
        
        # Sinks (nodes with only incoming edges)
        self.sinks = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0 and self.graph.in_degree(n) > 0]
        
        # Try to find cycles
        try:
            self.cycles = list(nx.simple_cycles(self.graph))
        except:
            self.cycles = []
    
    def _detect_hierarchies(self):
        """Detect hierarchical structures"""
        import networkx as nx
        
        # Check if graph is a tree or forest
        undirected = self.graph.to_undirected()
        self.is_tree = nx.is_tree(undirected) if len(undirected) > 0 else False
        self.is_forest = nx.is_forest(undirected) if len(undirected) > 0 else False
        
        # Try to do topological sort for DAGs
        try:
            self.topological_order = list(nx.topological_sort(self.graph))
            self.is_dag = True
        except:
            self.topological_order = []
            self.is_dag = False
    
    def _perform_spectral_analysis(self):
        """Perform spectral analysis on graph Laplacian"""
        import networkx as nx
        import numpy as np
        
        # Create Laplacian matrix
        try:
            undirected = self.graph.to_undirected()
            laplacian = nx.laplacian_matrix(undirected).todense()
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(laplacian)
            
            # Store spectral properties
            self.spectral_gap = eigenvalues[1] if len(eigenvalues) > 1 else 0
            self.algebraic_connectivity = self.spectral_gap  # Same as second smallest eigenvalue
            self.spectral_radius = eigenvalues[-1] if len(eigenvalues) > 0 else 0
        except:
            self.spectral_gap = 0
            self.algebraic_connectivity = 0
            self.spectral_radius = 0
    
    def _compile_results(self):
        """Compile analysis results"""
        # Basic metrics
        metrics = {
            "node_count": len(self.graph),
            "edge_count": self.graph.size(),
            "density": nx.density(self.graph),
            "is_directed": nx.is_directed(self.graph),
            "is_tree": self.is_tree,
            "is_forest": self.is_forest,
            "is_dag": self.is_dag,
            "components": len(self.components),
            "diameter": self._safe_diameter(),
            "average_clustering": self._safe_clustering(),
            "has_cycles": len(self.cycles) > 0,
            "spectral_gap": self.spectral_gap,
            "algebraic_connectivity": self.algebraic_connectivity
        }
        
        # Node importance
        if self.eigenvector_centralities:
            metrics["most_central_nodes"] = sorted(
                self.eigenvector_centralities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        
        # Community structure
        if self.spectral_clusters:
            metrics["community_structure"] = [
                {"id": f"cluster_{i}", "size": len(cluster), "nodes": cluster}
                for i, cluster in enumerate(self.spectral_clusters)
            ]
        
        # Flow structure for directed graphs
        if nx.is_directed(self.graph):
            metrics["sources"] = self.sources
            metrics["sinks"] = self.sinks
            metrics["cycles"] = self.cycles[:5]  # Limit to first 5 cycles
            if self.topological_order:
                metrics["topological_levels"] = self._identify_topological_levels()
        
        return metrics
    
    def _safe_diameter(self):
        """Calculate diameter safely"""
        import networkx as nx
        try:
            undirected = self.graph.to_undirected()
            if nx.is_connected(undirected):
                return nx.diameter(undirected)
            else:
                return -1  # Indicates disconnected graph
        except:
            return -1
    
    def _safe_clustering(self):
        """Calculate clustering coefficient safely"""
        import networkx as nx
        try:
            return nx.average_clustering(self.graph.to_undirected())
        except:
            return 0
    
    def _identify_topological_levels(self):
        """Group nodes by their topological level"""
        if not self.topological_order:
            return []
            
        levels = {}
        visited = set()
        
        for node in self.topological_order:
            # Node's level is 1 + maximum level of its predecessors
            pred_levels = [levels.get(pred, 0) for pred in self.graph.predecessors(node) if pred in visited]
            level = 1 + (max(pred_levels) if pred_levels else 0)
            levels[node] = level
            visited.add(node)
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        return [{"level": k, "nodes": v} for k, v in sorted(level_groups.items())]i

class ProbabilisticDiagramConnectionModel:
    """Monte Carlo probability model for handling uncertain diagram element connections"""
    
    def __init__(self, elements, confidence_threshold=0.65):
        self.elements = elements
        self.confidence_threshold = confidence_threshold
        self.connection_probabilities = {}
        self.connection_samples = []
        self.num_samples = 1000
    
    def compute_connection_probabilities(self):
        """Compute probabilities of connections between all element pairs"""
        import numpy as np
        
        # Create lookup indices for fast access
        element_indices = {elem.id: i for i, elem in enumerate(self.elements)}
        
        # Initialize probability matrix
        n = len(self.elements)
        probability_matrix = np.zeros((n, n))
        
        # Calculate connection probabilities for each element pair
        for i, source in enumerate(self.elements):
            for j, target in enumerate(self.elements):
                if i == j:
                    continue  # Skip self-connections
                
                # Skip obvious non-connections
                if source.type in ["line", "arrow"] or target.type in ["line", "arrow"]:
                    continue
                
                # Compute probability based on geometric and semantic features
                probability = self._compute_connection_probability(source, target)
                probability_matrix[i, j] = probability
                
                if probability > 0.1:  # Only store significant probabilities
                    self.connection_probabilities[(source.id, target.id)] = probability
        
        # Generate Monte Carlo samples
        self._generate_connection_samples(probability_matrix)
        
        return self.connection_probabilities
    
    def _compute_connection_probability(self, source, target):
        """Compute connection probability between two elements"""
        import numpy as np
        
        # Base probability
        probability = 0.0
        
        # Spatial proximity factors
        if source.type not in ["line", "arrow"] and target.type not in ["line", "arrow"]:
            # Get element centers
            sx, sy = source.center()
            tx, ty = target.center()
            
            # Calculate distance
            distance = np.sqrt((sx - tx)**2 + (sy - ty)**2)
            
            # Calculate sizes
            source_size = max(source.width, source.height) if hasattr(source, 'width') else 0
            target_size = max(target.width, target.height) if hasattr(target, 'width') else 0
            avg_size = (source_size + target_size) / 2 if source_size + target_size > 0 else 50
            
            # Distance factor - probability decreases with distance
            distance_factor = np.exp(-distance / (3 * avg_size))
            probability += 0.4 * distance_factor
        
        # Element type compatibility
        type_compatibility = self._get_type_compatibility(source.type, target.type)
        probability += 0.3 * type_compatibility
        
        # Text similarity (for classes, entities, etc.)
        if hasattr(source, 'text') and hasattr(target, 'text') and source.text and target.text:
            text_similarity = self._compute_text_similarity(source.text, target.text)
            probability += 0.2 * text_similarity
        
        # Check if there's any connecting lines or arrows
        # This would be determined by looking at the elements array for lines/arrows
        # connecting these elements, but for simplicity we'll skip this computation
        
        # Clamp probability to [0, 1]
        return max(0.0, min(1.0, probability))
    
    def _get_type_compatibility(self, type1, type2):
        """Get compatibility score between element types"""
        # Define compatibility matrix for common element types
        compatibility = {
            "rectangle": {
                "rectangle": 0.5,
                "diamond": 0.7,
                "ellipse": 0.3,
                "circle": 0.3
            },
            "diamond": {
                "rectangle": 0.7,
                "diamond": 0.2,
                "ellipse": 0.1,
                "circle": 0.1
            },
            "ellipse": {
                "rectangle": 0.3,
                "diamond": 0.1,
                "ellipse": 0.2,
                "circle": 0.6
            },
            "circle": {
                "rectangle": 0.3,
                "diamond": 0.1,
                "ellipse": 0.6,
                "circle": 0.2
            }
        }
        
        # Get compatibility score or default to 0.1
        return compatibility.get(type1, {}).get(type2, 0.1)
    
    def _compute_text_similarity(self, text1, text2):
        """Compute similarity between two text strings"""
        import re
        
        # Tokenize texts
        def tokenize(text):
            return re.findall(r'\w+', text.lower())
        
        tokens1 = set(tokenize(text1))
        tokens2 = set(tokenize(text2))
        
        # Compute Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_connection_samples(self, probability_matrix):
        """Generate Monte Carlo samples of connection graphs"""
        import numpy as np
        import networkx as nx
        
        n = len(self.elements)
        
        for _ in range(self.num_samples):
            # Create a random adjacency matrix based on probabilities
            random_matrix = np.random.random((n, n))
            adjacency = (random_matrix < probability_matrix).astype(int)
            
            # Create a graph from the adjacency matrix
            G = nx.from_numpy_array(adjacency, create_using=nx.DiGraph)
            
            # Store the graph
            self.connection_samples.append(G)
    
    def get_most_likely_connections(self):
        """Get the most likely connections based on the probability threshold"""
        connections = []
        
        for (source_id, target_id), probability in self.connection_probabilities.items():
            if probability >= self.confidence_threshold:
                connections.append({
                    "source": source_id,
                    "target": target_id,
                    "probability": probability
                })
        
        return sorted(connections, key=lambda x: x["probability"], reverse=True)
    
    def get_connection_consensus(self):
        """Get connection consensus from Monte Carlo samples"""
        import numpy as np
        
        if not self.connection_samples:
            return []
        
        # Count connection occurrences across samples
        connection_counts = {}
        
        for g in self.connection_samples:
            for u, v in g.edges():
                source_id = self.elements[u].id
                target_id = self.elements[v].id
                key = (source_id, target_id)
                
                if key not in connection_counts:
                    connection_counts[key] = 0
                    
                connection_counts[key] += 1
        
        # Calculate empirical probabilities
        connections = []
        for (source_id, target_id), count in connection_counts.items():
            probability = count / self.num_samples
            if probability >= self.confidence_threshold:
                connections.append({
                    "source": source_id,
                    "target": target_id,
                    "probability": probability
                })
        
        return sorted(connections, key=lambda x: x["probability"], reverse=True)
