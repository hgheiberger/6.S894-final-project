#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable
{
public:
  sphere(const point3 &center, double radius, shared_ptr<material> obj_material)
      : center(center), radius(std::fmax(0, radius)), obj_material(obj_material) {}

  bool hit(const ray &r, interval valid_interval, hit_record &hit_rec) const override
  {
    // Solve ray sphere collision quadratic
    vec3 center_to_ray = center - r.origin();
    auto a = r.direction().length_squared();
    auto b = dot(r.direction(), center_to_ray);
    auto c = center_to_ray.length_squared() - radius * radius;
    auto discriminant = b * b - a * c;

    // If there are no solutions
    if (discriminant < 0)
      return false;

    // Gets closest ray collision solutions within valid range.
    auto root = (b - std::sqrt(discriminant)) / a;
    if (!valid_interval.surrounds(root))
    {
      root = (b + std::sqrt(discriminant)) / a;
      if (!valid_interval.surrounds(root))
        return false;
    }

    // Create hit_record based on found hit location
    hit_rec.obj_material = obj_material;
    hit_rec.t = root;
    hit_rec.p = r.point_at_parameter(hit_rec.t);
    vec3 outward_normal = (hit_rec.p - center) / radius;
    hit_rec.set_normal(r, outward_normal);

    return true;
  }

private:
  point3 center;
  double radius;
  shared_ptr<material> obj_material;
};

#endif