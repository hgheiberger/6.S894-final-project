#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"

class material
{
public:
  virtual ~material() = default;

  virtual bool scatter(
      const ray &r, const hit_record &rec, color &attenuation, ray &scattered) const
  {
    return false;
  }
};

class lambertian : public material
{
public:
  lambertian(const color &reflect_prob) : reflect_prob(reflect_prob) {}

  bool scatter(const ray &r, const hit_record &hit_rec, color &attenuation, ray &scattered_ray)
      const override
  {
    // Approximates Lambertian Reflection
    auto scatter_direction = hit_rec.normal + random_unit_vector();

    // Prevents impossible scatters from occuring
    if (scatter_direction.near_zero())
      scatter_direction = hit_rec.normal;

    // Outputs scattered ray
    scattered_ray = ray(hit_rec.p, scatter_direction);
    attenuation = reflect_prob;
    return true;
  }

private:
  color reflect_prob;
};

class metal : public material
{
public:
  metal(const color &reflect_prob, double fuzz_quantity) : reflect_prob(reflect_prob), fuzz_quantity(fuzz_quantity < 1 ? fuzz_quantity : 1) {}

  bool scatter(const ray &r, const hit_record &hit_rec, color &attenuation, ray &scattered_ray)
      const override
  {
    // Calculates metal refleciton
    vec3 reflected_ray = reflect(r.direction(), hit_rec.normal);

    // Adds fuzz to simuulate imperfections
    reflected_ray = unit_vector(reflected_ray) + (fuzz_quantity * random_unit_vector());

    // Outputs scattered ray
    scattered_ray = ray(hit_rec.p, reflected_ray);
    attenuation = reflect_prob;

    // Ensures reflected ray is in a possible direction
    return (dot(scattered_ray.direction(), hit_rec.normal) > 0);
  }

private:
  color reflect_prob;
  double fuzz_quantity;
};

class dielectric : public material
{
public:
  dielectric(double refraction_index) : refraction_index(refraction_index) {}

  bool scatter(const ray &r, const hit_record &hit_rec, color &attenuation, ray &scattered_ray)
      const override
  {
    vec3 r_unit_vector = unit_vector(r.direction());

    // If light ray is too glancing, Snell's law says it must reflect
    double ri = hit_rec.front_face ? (1.0 / refraction_index) : refraction_index;
    double cos_theta = std::fmin(dot(-r_unit_vector, hit_rec.normal), 1.0);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    bool will_reflect = ri * sin_theta > 1.0;

    // If too glancing or randomly chosen, reflect the ray
    vec3 direction;
    if (will_reflect || reflectance(cos_theta, ri) > random_double())
      direction = reflect(r_unit_vector, hit_rec.normal);
    // Otherwise, refract the ray
    else
      direction = refract(r_unit_vector, hit_rec.normal, ri);

    // Outputs scattered ray
    scattered_ray = ray(hit_rec.p, direction);
    attenuation = color(1.0, 1.0, 1.0);
    return true;
  }

private:
  double refraction_index;

  static double reflectance(double cosine, double refraction_index)
  {
    // Calculates reflectance using Schlick's approximation
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * std::pow((1 - cosine), 5);
  }
};

#endif