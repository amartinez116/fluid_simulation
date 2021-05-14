#include <cstdio>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "fluid_cuda.cuh"
#include "fluid.hpp"

__device__ void 
computeDensity(int index, Particle* pi, const Particle* pj, float supportRadius) {
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
__device__ void 
computePressure(Particle* p) {
    p->mPressure = GAS_STIFFNESS * (p->mDensity - REST_DENSITY);
}

__device__ void 
computeViscosityForce(Particle *pi, Particle *pj, float supportRadius) {
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

__device__ void 
computePressureForce(int index, Particle *pi, const Particle *pj, float supportRadius) {
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

__device__ void 
employEulerIntegrator(Particle *p, const float x, const float y, const float z) {
    p->mAcceleration.x = x / p->mDensity;
    p->mAcceleration.y = y / p->mDensity;
    p->mAcceleration.z = z / p->mDensity;

    p->mVelocity.x = p->mVelocity.x + p->mAcceleration.x * TIME_STEP;
    p->mVelocity.y = p->mVelocity.y + p->mAcceleration.y * TIME_STEP;
    p->mVelocity.z = p->mVelocity.z + p->mAcceleration.z * TIME_STEP;

    p->mPosition.x = p->mPosition.x + p->mVelocity.x * TIME_STEP;
    p->mPosition.y = p->mPosition.y + p->mVelocity.y * TIME_STEP;
    p->mPosition.z = p->mPosition.z + p->mVelocity.z * TIME_STEP;
}

__device__ bool 
detectCollision(Particle *p, float *contact_x, float *contact_y, float *contact_z,
                             float *normal_x, float *normal_y, float *normal_z) {
  if (abs(p->mPosition.x) <= BOX_SIZE / 2 &&
      abs(p->mPosition.y) <= BOX_SIZE / 2 &&
      abs(p->mPosition.z) <= BOX_SIZE / 2)
    return false;

  char maxComponent = 'x';
  float maxValue = abs(p->mPosition.x);
  if (maxValue < abs(p->mPosition.y)) {
    maxComponent = 'y';
    maxValue = abs(p->mPosition.y);
  }
  if (maxValue < abs(p->mPosition.z)) {
    maxComponent = 'z';
    maxValue = abs(p->mPosition.z);
  }
  // 'unitSurfaceNormal' is based on the current position component with the
  // largest absolute value
  switch (maxComponent) {
    case 'x':
        if (p->mPosition.x < -BOX_SIZE / 2) {
            *contact_x = -BOX_SIZE / 2;
            *contact_y = p->mPosition.y;
            *contact_z = p->mPosition.z;

            if (p->mPosition.y < -BOX_SIZE / 2) {
                *contact_y = -BOX_SIZE / 2;
            } else if (p->mPosition.y > BOX_SIZE / 2) {
                *contact_y = BOX_SIZE / 2;
            }

            if (p->mPosition.z < -BOX_SIZE / 2) {
                *contact_z = -BOX_SIZE / 2;
            } else if (p->mPosition.z > BOX_SIZE / 2) {
                *contact_z = BOX_SIZE / 2;
            }

            *normal_x = 1.0f;
            *normal_y = 0.0f;
            *normal_z = 0.0f;
        } else if (p->mPosition.x > BOX_SIZE / 2) {
            *contact_x = BOX_SIZE / 2;
            *contact_y = p->mPosition.y;
            *contact_z = p->mPosition.z;

            if (p->mPosition.y < -BOX_SIZE / 2) {
                *contact_y = -BOX_SIZE / 2;
            } else if (p->mPosition.y > BOX_SIZE / 2) {
                *contact_y = BOX_SIZE / 2;
            }
    
            if (p->mPosition.z < -BOX_SIZE / 2) {
                *contact_z = -BOX_SIZE / 2;
            } else if (p->mPosition.z > BOX_SIZE / 2) {
                *contact_z = BOX_SIZE / 2;
            }
    
            *normal_x = -1.0f;
            *normal_y = 0.0f;
            *normal_z = 0.0f;
        }
        break;
    case 'y':
        if (p->mPosition.y < -BOX_SIZE / 2) {
            *contact_x = p->mPosition.x;
            *contact_y = -BOX_SIZE / 2;
            *contact_z = p->mPosition.z;

            if (p->mPosition.x < -BOX_SIZE / 2) {
                *contact_x = -BOX_SIZE / 2;
            } else if (p->mPosition.x > BOX_SIZE / 2) {
                *contact_x = BOX_SIZE / 2;
            }

            if (p->mPosition.z < -BOX_SIZE / 2) {
                *contact_z = -BOX_SIZE / 2;
            } else if (p->mPosition.z > BOX_SIZE / 2) {
                *contact_z = BOX_SIZE / 2;
            }

            *normal_x = 0.0f;
            *normal_y = 1.0f;
            *normal_z = 0.0f;
        } else if (p->mPosition.y > BOX_SIZE / 2) {
            *contact_x = p->mPosition.x;
            *contact_y = BOX_SIZE / 2;
            *contact_z = p->mPosition.z;

            if (p->mPosition.x < -BOX_SIZE / 2) {
                *contact_x = -BOX_SIZE / 2;
            } else if (p->mPosition.x > BOX_SIZE / 2) {
                *contact_x = BOX_SIZE / 2;
            }

            if (p->mPosition.z < -BOX_SIZE / 2) {
                *contact_z = -BOX_SIZE / 2;
            } else if (p->mPosition.z > BOX_SIZE / 2) {
                *contact_z = BOX_SIZE / 2;
            }

            *normal_x = 0.0f;
            *normal_y = -1.0f;
            *normal_z = 0.0f;
        }
        break;
  case 'z':
        if (p->mPosition.z < -BOX_SIZE / 2) {
            *contact_x = p->mPosition.x;
            *contact_y = p->mPosition.y;
            *contact_z = -BOX_SIZE / 2;

            if (p->mPosition.x < -BOX_SIZE / 2) {
                *contact_x = -BOX_SIZE / 2;
            } else if (p->mPosition.x > BOX_SIZE / 2) {
                *contact_x = BOX_SIZE / 2;
            }

            if (p->mPosition.y < -BOX_SIZE / 2) {
                *contact_y = -BOX_SIZE / 2;
            } else if (p->mPosition.y > BOX_SIZE / 2) {
                *contact_y = BOX_SIZE / 2;
            }

            *normal_x = 0.0f;
            *normal_y = 0.0f;
            *normal_z = 1.0f;
        } else if (p->mPosition.z > BOX_SIZE / 2) {
            *contact_x = p->mPosition.x;
            *contact_y = p->mPosition.y;
            *contact_z = BOX_SIZE / 2;

            if (p->mPosition.x < -BOX_SIZE / 2) {
                *contact_x = -BOX_SIZE / 2;
            } else if (p->mPosition.x > BOX_SIZE / 2) {
                *contact_x = BOX_SIZE / 2;
            }

            if (p->mPosition.y < -BOX_SIZE / 2) {
                *contact_y = -BOX_SIZE / 2;
            } else if (p->mPosition.y > BOX_SIZE / 2) {
                *contact_y = BOX_SIZE / 2;
            }

            *normal_x = 0.0f;
            *normal_y = 0.0f;
            *normal_z = -1.0f;
        }
        break;
  }
  return true;
}

__device__ void
updateVelocity(Particle *p, float contact_x, float contact_y, float contact_z,
                            float normal_x, float normal_y, float normal_z) {
    
    float p_x = p->mPosition.x - contact_x;
    float p_y = p->mPosition.y - contact_y;
    float p_z = p->mPosition.z - contact_z;
    float penetrationDepth = sqrt((p_x * p_x) + (p_y * p_y) + (p_z * p_z));

    float v_length = sqrt((p->mVelocity.x * p->mVelocity.x) + (p->mVelocity.y * p->mVelocity.y) + (p->mVelocity.z * p->mVelocity.z));
    float velocityDotNormal = (p->mVelocity.x * normal_x) + (p->mVelocity.y * normal_y) + (p->mVelocity.z * normal_z);
    p->mVelocity.x = p->mVelocity.x - normal_x * (1 + RESTITUTION * penetrationDepth / (TIME_STEP * v_length)) * velocityDotNormal;
    p->mVelocity.y = p->mVelocity.y - normal_y * (1 + RESTITUTION * penetrationDepth / (TIME_STEP * v_length)) * velocityDotNormal;
    p->mVelocity.z = p->mVelocity.z - normal_z * (1 + RESTITUTION * penetrationDepth / (TIME_STEP * v_length)) * velocityDotNormal;
}

__global__ void
calcDensityKernel(Particle *particles, int num_particles) {
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

__global__ void
calcInternalForcesKernel(Particle *particles, int num_particles) {
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

__global__ void
calcExternalForcesKernel(Particle *particles, int num_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_particles) {
        Particle *pi = &particles[index];
        pi->mGravitationalForce.x = 0.0f * pi->mDensity;
        pi->mGravitationalForce.y = GRAVITATIONAL_ACCELERATION * pi->mDensity;
        pi->mGravitationalForce.z = 0.0f * pi->mDensity;

        pi->mSurfaceNormal.x = 0.0f;
        pi->mSurfaceNormal.y = 0.0f;
        pi->mSurfaceNormal.z = 0.0f;
        for (int j = 0; j < num_particles; j++) {
            Particle *pj = &particles[j];
            float dist_x = pi->mPosition.x - pj->mPosition.x;
            float dist_y = pi->mPosition.y - pj->mPosition.y;
            float dist_z = pi->mPosition.z - pj->mPosition.z;

            float dist = sqrt((dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z));
            float volume = pj->mMass / pj->mDensity;
            if (dist <= SUPPORT_RADIUS) {
                float calc = (945 / (32 * M_PI * powf(SUPPORT_RADIUS, 9.0f))) * powf(SUPPORT_RADIUS * SUPPORT_RADIUS - dist * dist, 2.0f);

                pi->mSurfaceNormal.x +=  - (dist_x * calc) * volume;
                pi->mSurfaceNormal.y +=  - (dist_y * calc) * volume;
                pi->mSurfaceNormal.z +=  - (dist_z * calc) * volume;
            }
        }

        float normal_length = sqrt((pi->mSurfaceNormal.x * pi->mSurfaceNormal.x) + (pi->mSurfaceNormal.y * pi->mSurfaceNormal.y) + (pi->mSurfaceNormal.z * pi->mSurfaceNormal.z));

        pi->mSurfaceTensionForce.x = 0.0f;
        pi->mSurfaceTensionForce.y = 0.0f;
        pi->mSurfaceTensionForce.z = 0.0f;

        if (normal_length >= THRESHOLD) {
            for (int j = 0; j < num_particles; j++) {
                Particle *pj = &particles[j];
                float dist_x = pi->mPosition.x - pj->mPosition.x;
                float dist_y = pi->mPosition.y - pj->mPosition.y;
                float dist_z = pi->mPosition.z - pj->mPosition.z;
    
                float dist = sqrt((dist_x * dist_x) + (dist_y * dist_y) + (dist_z * dist_z));
                float volume = pj->mMass / pj->mDensity;
                if (dist <= SUPPORT_RADIUS) {
                    float calc = -(945 / (32 * M_PI * powf(SUPPORT_RADIUS, 9.0f))) *
                                  (SUPPORT_RADIUS * SUPPORT_RADIUS - dist * dist) *
                                  (3 * SUPPORT_RADIUS * SUPPORT_RADIUS - 7 * dist * dist);

                    pi->mSurfaceTensionForce.x += volume * calc;
                    pi->mSurfaceTensionForce.y += volume * calc;
                    pi->mSurfaceTensionForce.z += volume * calc;
                }
            }

            pi->mSurfaceTensionForce.x = - (pi->mSurfaceNormal.x / normal_length) * SURFACE_TENSION * pi->mSurfaceTensionForce.x;
            pi->mSurfaceTensionForce.y = - (pi->mSurfaceNormal.y / normal_length) * SURFACE_TENSION * pi->mSurfaceTensionForce.y;
            pi->mSurfaceTensionForce.z = - (pi->mSurfaceNormal.z / normal_length) * SURFACE_TENSION * pi->mSurfaceTensionForce.z;
        }
    }
}

__global__ void
handleCollisionKernel(Particle* particles, int num_particles) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < num_particles) {
        Particle *pi = &particles[index];
        float force_x = pi->mPressureForce.x + pi->mViscosityForce.x + 
                        pi->mGravitationalForce.x + pi->mSurfaceTensionForce.x;
        float force_y = pi->mPressureForce.y + pi->mViscosityForce.y + 
                        pi->mGravitationalForce.y + pi->mSurfaceTensionForce.y;
        float force_z = pi->mPressureForce.z + pi->mViscosityForce.z + 
                        pi->mGravitationalForce.z + pi->mSurfaceTensionForce.z;
        employEulerIntegrator(pi, force_x, force_y, force_z);

        float contact_x;
        float contact_y;
        float contact_z;
        float normal_x;
        float normal_y;
        float normal_z;

        if (detectCollision(pi, &contact_x, &contact_y, &contact_z, &normal_x, &normal_y, &normal_z)) {
            updateVelocity(pi, contact_x, contact_y, contact_z, normal_x, normal_y, normal_z);
            pi->mPosition.x = contact_x;
            pi->mPosition.y = contact_y;
            pi->mPosition.z = contact_z;
        }
    }
}

void cudaComputeDensities(Particle* cudaParticles, int num_particles) {
    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    calcDensityKernel <<<blocks, threadsPerBlock>>>(cudaParticles, num_particles);
    cudaDeviceSynchronize();
}

void cudaComputeInternalForces(Particle *cudaParticles, int num_particles) {
    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    calcInternalForcesKernel <<<blocks, threadsPerBlock>>>(cudaParticles, num_particles);
    cudaDeviceSynchronize();
}

void cudaComputeExternalForces(Particle *cudaParticles, int num_particles) {
    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    calcExternalForcesKernel <<<blocks, threadsPerBlock>>>(cudaParticles, num_particles);
    cudaDeviceSynchronize();
}

void cudaCollisionHandling(Particle* cudaParticles, int num_particles) {
    const int threadsPerBlock = 128;
    const int blocks = (num_particles + threadsPerBlock - 1) / threadsPerBlock;

    handleCollisionKernel <<<blocks, threadsPerBlock>>> (cudaParticles, num_particles);
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