#ifndef RAYH
#define RAYH

#include "vec3.h"

class ray
{
    public:
        __device__ ray() {}

        __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction) {}
        
        __device__ vec3 origin() const       { return orig; }
        __device__ vec3 direction() const    { return dir; }
        
        __device__ vec3 point_at_parameter(float t) const { return orig + t*dir; }

    private:
        point3 orig;
        vec3 dir;
};

#endif