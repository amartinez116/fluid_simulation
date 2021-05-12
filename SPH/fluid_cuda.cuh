#include "particle.hpp"

extern "C"
{
    void cudaComputeDensities(Particle* host_particles, int num_particles, Particle* result_particles);

    void allocateArray(void** devPtr, size_t size);

    void copyArrayToDevice(void* device, const void* host, size_t size);

    void copyArrayFromDevice(void* host, const void* device, size_t size);
}