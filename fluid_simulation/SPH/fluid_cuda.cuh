#include "particle.hpp"

extern "C"
{
    void cudaComputeDensities(Particle* cudaParticles, int num_particles);
    void cudaComputeInternalForces(Particle* cudaParticles, int num_particles);
    void cudaComputeExternalForces(Particle* cudaParticles, int num_particles);
    void cudaCollisionHandling(Particle* cudaParticles, int num_particles);

    void allocateArray(void** devPtr, size_t size);
    void copyArrayToDevice(void* device, const void* host, size_t size);
    void copyArrayFromDevice(void* host, const void* device, size_t size);
}
