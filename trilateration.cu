#include <iostream>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>

#define THRESH 1.0f

__device__ float2 operator /= (float2 quotient, float divisor) {
    quotient.x /= divisor;
    quotient.y /= divisor;
    return quotient;
}
bool operator == (const float2 first, const float2 second) {
    return (fabs(first.y-second.y) <= THRESH)&&(fabs(first.x-second.x) <= THRESH);
}
__device__ float2 operator += (float2 first,float2 other) {
    first.x += other.x;
    first.y += other.y;
    return first;
}

float euclidian_distance(const float2 p1,const float2 p2) {
    return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}

__global__ void trilateration(float2*results,float3*distances,float2 guard_point_a,float2 guard_point_b,float2 guard_point_c){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float2 p1 = guard_point_a;
    float d1 = distances[idx].x;
    float2 p2 = guard_point_b;
    float d2 = distances[idx].y;
    float2 p3 = guard_point_c;
    float d3 = distances[idx].z;
    float numerator = ((p2.x - p1.x)*(p3.x * p3.x + p3.y * p3.y - d3 * d3) +
                       (p1.x - p3.x)*(p2.x * p2.x + p2.y * p2.y - d2 * d2) +
                       (p3.x - p2.x)*(p1.x * p1.x + p1.y * p1.y - d1 * d1));
    float denominator = (2 * (p3.y*(p2.x - p1.x) + p2.y*(p1.x - p3.x) + p1.y*(p3.x - p2.x)));
    float y = numerator/denominator;
    float x = (d2 * d2 + p1.x * p1.x + p1.y * p1.y - d1 * d1 - p2.x * p2.x - p2.y * p2.y
               -2*(p1.y - p2.y)*y)/
    (2*(p1.x - p2.x));
    float2 result = {x,y};
    results[idx] = result;
}
__global__ void average(float2 * results, float2 * averages) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    averages[idx].x = 0;
    averages[idx].y = 0;
    averages[idx] = results[4*idx];
    averages[idx] = results[4*idx+1];
    averages[idx] = results[4*idx+2];
    averages[idx] = results[4*idx+3];
    averages[idx] /= 4.0;
}
void generate_data(int NUM) {
    assert(NUM%4 ==0);
    float3 * distances = new float3[NUM];
    float2 * trail = new float2[NUM];
    float2 * results = new float2[NUM];
    float3 * device_distances;
    float2 * device_results;
    float2 * device_averages;
    float2 * averages = new float2[NUM/4];
    srand (time(NULL));
    const float2 guard_point_a = {(float)rand()/(float)RAND_MAX*1000.0,(float)rand()/(float)RAND_MAX*1000.0};
    const float2 guard_point_b = {(float)rand()/(float)RAND_MAX*1000.0,(float)rand()/(float)RAND_MAX*1000.0};
    const float2 guard_point_c = {(float)rand()/(float)RAND_MAX*1000.0,(float)rand()/(float)RAND_MAX*1000.0};
    float2 point = {(float)rand()/(float)RAND_MAX*500.0,(float)rand()/(float)RAND_MAX*500.0};
    trail[0] = point;
    distances[0].x = euclidian_distance(point,guard_point_a);
    distances[0].y = euclidian_distance(point,guard_point_b);
    distances[0].z = euclidian_distance(point,guard_point_c);
    for (int i = 1; i < NUM; i++) {
        point = trail[i-1];
        point.x += (float)rand()/(float)RAND_MAX;
        point.y += (float)rand()/(float)RAND_MAX;
        trail[i] = point;
        distances[i].x = euclidian_distance(point,guard_point_a);
        distances[i].y = euclidian_distance(point,guard_point_b);
        distances[i].z = euclidian_distance(point,guard_point_c);
    }
    size_t distances_size = NUM * sizeof(float3);
    size_t results_size = NUM * sizeof(float2);
    cudaMalloc((void**)&device_distances,distances_size);
    cudaMalloc((void**)&device_results,results_size);
    cudaMalloc((void**)&device_averages,results_size/4);
    cudaMemcpy(device_distances,distances,distances_size,cudaMemcpyHostToDevice);
    int block_num = 4;
    trilateration<<<block_num,NUM/block_num>>>(device_results,device_distances,guard_point_a,guard_point_b,guard_point_c);
    average<<<block_num,NUM/block_num/4>>>(device_results,device_averages);
    for (int i = 0; i < NUM; i++) {
        results[i].x=0;
        results[i].y=0;
    }
    cudaMemcpy(results,device_results,results_size,cudaMemcpyDeviceToHost);
    cudaMemcpy(averages,device_averages,results_size/4,cudaMemcpyDeviceToHost);
    for (int i = 0; i < NUM;i++) {
        std::cout << "x: " << results[i].x << ", y: " << results[i].y << "			";
        std::cout << "x: " << trail[i].x << ", y: " << trail[i].y << "\n";
        std::cout << (results[i] == trail[i] ? "Pass\n" : "Fail\n");
    }
    for (int i = 0; i < NUM/4; i++) {
        std::cout << "average x: " << averages[i].x << " average y: " << averages[i].y << std::endl;
    }
    delete [] distances;
    delete [] trail;
    delete [] results;
    delete [] averages;
    cudaFree(device_averages);
    cudaFree(device_distances);
    cudaFree(device_results);
}

int main() {
    //clock_t start = clock();
    //double diff;
    generate_data(4096);
    //diff = ( std::clock() - start ) / (double)CLOCKS_PER_SEC;
    //std::cout << "Time elapsed: "<< diff <<'\n';
    return 0;
}
