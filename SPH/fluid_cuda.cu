#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "fluid_cuda.cuh"
#include "fluid.hpp"

__device__ void computeDensity(Particle* pi, const Particle* pj, float supportRadius) {
    const float POLY6 = 315.f / (64.f * M_PI * powf(supportRadius, 9.f));
    float x = pi->mPosition.x - pj->mPosition.x;
    float y = pi->mPosition.y - pj->mPosition.y;
    float z = pi->mPosition.z - pj->mPosition.z;

    float r2 = powf(x*x + y*y + z*z, 0.5);

    if (r2 < supportRadius) {
        pi->mDensity += POLY6 * powf(supportRadius - r2, 3.f);
    }
}

__device__ void computePressure(Particle* p) {
    p->mPressure = fmaxf(0.f, GAS_STIFFNESS * (p->mDensity - REST_DENSITY));
}

__global__ void
calcDensitykernel(Particle* particles, int num_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_particles) {
        Particle *pi = &particles[index];
        pi->mDensity = 0.0f;
        for (int j = 0; j < num_particles; j++) {
            Particle *pj = &particles[j];
            computeDensity(pi, pj, 0);
        }
        computePressure(pi);
    }
}

void cudaComputeDensities(Particle* host_particles, int num_particles) {
    Particle* device_particles;
    cudaMalloc((void **) &device_particles, sizeof(Particle) * num_particles);
    cudaMemcpy(device_particles, host_particles, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    calcDensitykernel <<<blocks, threadsPerBlock>>>(device_particles, num_particles);

    cudaDeviceSynchronize();

    cudaMemcpy(host_particles, device_particles, num_particles * sizeof(Particle), cudaMemcpyDeviceToHost);
}

void allocateArray(void** devPtr, size_t size) {
    cudaMalloc(devPtr, size);
}

void copyArrayToDevice(void* device, const void* host, size_t size) {
    cudaMemcpy((char*)device, host, size, cudaMemcpyHostToDevice);
}