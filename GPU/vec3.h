#ifndef VEC3H
#define VEC3H

#include <curand_kernel.h>

#define RANDOM_VEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

class vec3
{

public:
    float e[3];

    __host__ __device__ vec3() : e{0, 0, 0} {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }

    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float &operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3 &operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline vec3 &operator/=(const float t)
    {
        return *this *= 1.0 / t;
    }

    __host__ __device__ inline vec3 &operator*=(const vec3 &v);

    __host__ __device__ inline float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline float length() const { return sqrt(length_squared()); }
};

using point3 = vec3;

inline std::ostream &operator<<(std::ostream &out, const vec3 &vec)
{
    out << vec.e[0] << " " << vec.e[1] << " " << vec.e[2];
    return out;
}

__host__ __device__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
    return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
    return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2.0f * dot(v, n) * n;
}

__device__ bool refract(const vec3 &v, const vec3 &n, float etai_over_etat, vec3 &refracted)
{
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - etai_over_etat * etai_over_etat * (1 - dt * dt);
    if (discriminant > 0)
    {
        refracted = etai_over_etat * (uv - n * dt) - n * sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vec3 random_unit_vector(curandState *local_rand_state)
{
    vec3 p = 2.0f * RANDOM_VEC3 - vec3(1, 1, 1);
    while (p.length_squared() >= 1.0f)
    {
        p = 2.0f * RANDOM_VEC3 - vec3(1, 1, 1);
    }
    return p;
}

#endif