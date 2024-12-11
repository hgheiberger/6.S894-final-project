#ifndef MATERIALH
#define MATERIALH

#include "hittable.h"

class material
{
public:
  __device__ virtual bool scatter(
      const ray &r_in,
      const hit_record &rec,
      vec3 &attenuation,
      ray &scattered,
      curandState *local_rand_state) const
  {
    return false;
  }
};

class lambertian : public material
{
  // albedo := Fraction of light that a surface reflects
public:
  __device__ lambertian(const color &albedo) : albedo(albedo) {}

  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, color &attenuation, ray &scattered, curandState *local_rand_state) const
  {
    vec3 scatter_direction = rec.normal + random_unit_vector(local_rand_state);

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  }

  vec3 albedo;
};

class metal : public material
{
public:
  __device__ metal(const vec3 &albedo, float fuzz) : albedo(albedo)
  {
    if (fuzz < 1)
      fuzz = fuzz;
    else
      fuzz = 1;
  }
  __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
  {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * random_unit_vector(local_rand_state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
  }

  vec3 albedo;
  float fuzz;
};

class dielectric : public material
{
public:
  __device__ dielectric(float ri) : reflect_index(ri) {}
  __device__ virtual bool scatter(const ray &r_in,
                                  const hit_record &rec,
                                  vec3 &attenuation,
                                  ray &scattered,
                                  curandState *local_rand_state) const
  {
    vec3 out_normal;
    float etai_over_etat;
    vec3 refracted;
    float reflect_prob;
    float cosine;

    vec3 reflected = reflect(r_in.direction(), rec.normal);
    attenuation = vec3(1.0, 1.0, 1.0);
    if (dot(r_in.direction(), rec.normal) > 0.0f)
    {
      etai_over_etat = reflect_index;
      cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
      cosine = sqrt(1.0f - reflect_index * reflect_index * (1 - cosine * cosine));
      out_normal = -rec.normal;
    }
    else
    {
      etai_over_etat = 1.0f / reflect_index;
      cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
      out_normal = rec.normal;
    }

    // Calculates the probability of reflection based on the angle of the incoming ray
    if (refract(r_in.direction(), out_normal, etai_over_etat, refracted))
      reflect_prob = reflectance(cosine, reflect_index);
    else
      reflect_prob = 1.0f;

    // Determines whether the ray is randomly reflected instead of refracted
    if (curand_uniform(local_rand_state) < reflect_prob)
      scattered = ray(rec.p, reflected);
    else
      scattered = ray(rec.p, refracted);
    return true;
  }

  float reflect_index;

private:
  // Uses Schlick's reflectance approximation
  __device__ static float reflectance(float cosine, float reflect_index)
  {
    float r0 = (1.0f - reflect_index) / (1.0f + reflect_index);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
  }
};
#endif