#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "fluid_cuda.cuh"
#include "fluid.hpp"

__device__ void computeDensity(int index, Particle *pi, const Particle *pj, float supportRadius) {
    const float POLY6 = 315 / (64 * M_PI * powf(supportRadius, 9.0f));
    float x = pi->mPosition.x - pj->mPosition.x;
    float y = pi->mPosition.y - pj->mPosition.y;
    float z = pi->mPosition.z - pj->mPosition.z;

    float r2 = sqrt((x * x) + (y * y) + (z * z));

    if (r2 < supportRadius) {
        pi->mDensity += pj->mMass * POLY6 * powf(supportRadius * supportRadius - r2 * r2, 3.0f);
    }
}

// TODO: do we include the fmax? we get diff results w/ it
__device__ void computePressure(Particle *p) {
    p->mPressure = GAS_STIFFNESS * (p->mDensity - REST_DENSITY);
}

__global__ void
calcDensitykernel(Particle *particles, int num_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_particles) {
        Particle *pi = &particles[index];
        pi->mDensity = 0.0f;
        for (int j = 0; j < num_particles; j++) {
            Particle *pj = &particles[j];
            computeDensity(index, pi, pj, SUPPORT_RADIUS);
        }
        computePressure(pi);
    }
}

void cudaComputeDensities(Particle *cudaParticles, int num_particles) {
    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    calcDensitykernel <<<blocks, threadsPerBlock>>>(cudaParticles, num_particles);

    cudaDeviceSynchronize();
}

__device__ void computePressureForce(int index, Particle *pi, const Particle *pj, float supportRadius) {
    float x = pi->mPosition.x - pj->mPosition.x;
    float y = pi->mPosition.y - pj->mPosition.y;
    float z = pi->mPosition.z - pj->mPosition.z;
    float dist = sqrt((x * x) + (y * y) + (z * z));

    float density = pi->mDensity;
    float pressure = pi->mPressure;

    float distX = x / dist;
    float distY = y / dist;
    float distZ = z / dist;

    float valX = 0.0f;
    float valY = 0.0f;
    float valZ = 0.0f;

    if (dist <= supportRadius) {
        if (dist < 10e-5) { // If ||r|| -> 0+
            valX = -(1.0f / sqrt(3.0f)) * (45 / (M_PI * powf(supportRadius, 6.0f))) * powf(supportRadius - dist, 2.0f);
            valY = -(1.0f / sqrt(3.0f)) * (45 / (M_PI * powf(supportRadius, 6.0f))) * powf(supportRadius - dist, 2.0f);
            valZ = -(1.0f / sqrt(3.0f)) * (45 / (M_PI * powf(supportRadius, 6.0f))) * powf(supportRadius - dist, 2.0f);
        } else {
            valX = -distX * (45 / (M_PI * powf(supportRadius, 6.0f))) * powf(supportRadius - dist, 2.0f);
            valY = -distY * (45 / (M_PI * powf(supportRadius, 6.0f))) * powf(supportRadius - dist, 2.0f);
            valZ = -distZ * (45 / (M_PI * powf(supportRadius, 6.0f))) * powf(supportRadius - dist, 2.0f);
        }

        float pressureVal = (pressure / (density * density) +
                             pj->mPressure /
                             (pj->mDensity * pj->mDensity)) * pj->mMass;

        pi->mPressureForce.x += valX * pressureVal;
        pi->mPressureForce.y += valY * pressureVal;
        pi->mPressureForce.z += valZ * pressureVal;
    }
}

__device__ void computeViscosityForce(Particle *pi, Particle *pj, float supportRadius) {
    float x = pi->mPosition.x - pj->mPosition.x;
    float y = pi->mPosition.y - pj->mPosition.y;
    float z = pi->mPosition.z - pj->mPosition.z;
    float dist = sqrt((x * x) + (y * y) + (z * z));

    float useViscosityKernel_laplacian = 0.0f;

    if (dist <= supportRadius) {
        useViscosityKernel_laplacian = (45 / (M_PI * powf(supportRadius, 6.0f))) * (supportRadius - dist);
    }

    pi->mViscosityForce.x += (pj->mVelocity.x - pi->mVelocity.x) *
                             (pj->mMass / pj->mDensity) * useViscosityKernel_laplacian;

    pi->mViscosityForce.y += (pj->mVelocity.y - pi->mVelocity.y) *
                             (pj->mMass / pj->mDensity) * useViscosityKernel_laplacian;

    pi->mViscosityForce.z += (pj->mVelocity.z - pi->mVelocity.z) *
                             (pj->mMass / pj->mDensity) * useViscosityKernel_laplacian;
}

__global__ void
calcInternalForceskernel(Particle *particles, int num_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_particles) {
        Particle *pi = &particles[index];
        pi->mPressureForce.x = 0.0f;
        pi->mPressureForce.y = 0.0f;
        pi->mPressureForce.z = 0.0f;

        pi->mViscosityForce.x = 0.0f;
        pi->mViscosityForce.y = 0.0f;
        pi->mViscosityForce.z = 0.0f;

        for (int j = 0; j < num_particles; j++) {
            Particle *pj = &particles[j];
            if (index != j) {
                computePressureForce(index, pi, pj, SUPPORT_RADIUS);
                computeViscosityForce(pi, pj, SUPPORT_RADIUS);
            }
        }
        pi->mPressureForce.x = -(pi->mPressureForce.x * pi->mDensity);
        pi->mPressureForce.y = -(pi->mPressureForce.y * pi->mDensity);
        pi->mPressureForce.z = -(pi->mPressureForce.z * pi->mDensity);

        pi->mViscosityForce.x = pi->mViscosityForce.x * VISCOSITY;
        pi->mViscosityForce.y = pi->mViscosityForce.y * VISCOSITY;
        pi->mViscosityForce.z = pi->mViscosityForce.z * VISCOSITY;
    }
}

void cudaComputeInternalForces(Particle *cudaParticles, int num_particles) {
    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    calcInternalForceskernel <<<blocks, threadsPerBlock>>>(cudaParticles, num_particles);

    cudaDeviceSynchronize();
}

void cudaComputeExternalForces(Particle *cudaParticles, int num_particles) {
    cudaDeviceSynchronize();
}

void cudaCollisionHandling(Particle *cudaParticles, int num_particles) {
    cudaDeviceSynchronize();
}

void allocateArray(void **devPtr, size_t size) {
    cudaMalloc(devPtr, size);
}

void copyArrayToDevice(void *device, const void *host, size_t size) {
    cudaMemcpy((char *) device, host, size, cudaMemcpyHostToDevice);
}

void copyArrayFromDevice(void *host, const void *device, size_t size) {
    cudaMemcpy((char *) host, device, size, cudaMemcpyDeviceToHost);
}