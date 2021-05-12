#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "fluid_cuda.cuh"
#include "fluid.hpp"

__device__ void computeDensity(Particle* pi, const Particle* pj, float supportRadius) {
    const float POLY6 = 315 / (64 * M_PI * powf(supportRadius, 9.0f));
    float x = pi->mPosition.x - pj->mPosition.x;
    float y = pi->mPosition.y - pj->mPosition.y;
    float z = pi->mPosition.z - pj->mPosition.z;

    float r2 = sqrt((x * x) + (y * y) + (z * z));

    if (r2 < supportRadius) {
        pi->mDensity += pj->mMass * POLY6 * powf(supportRadius*supportRadius - r2*r2, 3.0f);
    }
}

// TODO: do we include the fmax? we get diff results w/ it
__device__ void computePressure(Particle* p) {
    p->mPressure = GAS_STIFFNESS * (p->mDensity - REST_DENSITY);
}

__global__ void
calcDensitykernel(Particle* particles, int num_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_particles) {
        Particle *pi = &particles[index];
        pi->mDensity = 0.0f;
        for (int j = 0; j < num_particles; j++) {
            Particle *pj = &particles[j];
            computeDensity(pi, pj, SUPPORT_RADIUS);
        }
        computePressure(pi);
    }
}

void cudaComputeDensities(Particle* host_particles, int num_particles, Particle* result_particles) {
    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    calcDensitykernel <<<blocks, threadsPerBlock>>>(host_particles, num_particles);

    cudaDeviceSynchronize();

    copyArrayFromDevice(result_particles, host_particles, num_particles * sizeof(Particle));
}

void allocateArray(void** devPtr, size_t size) {
    cudaMalloc(devPtr, size);
}

void copyArrayToDevice(void* device, const void* host, size_t size) {
    cudaMemcpy((char*)device, host, size, cudaMemcpyHostToDevice);
}

void copyArrayFromDevice(void* host, const void* device, size_t size){
    cudaMemcpy((char*)host, device, size, cudaMemcpyDeviceToHost);
}