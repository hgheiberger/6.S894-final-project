#ifndef HITTABLE_H
#define HITTABLE_H

class material;

class hit_record
{
public:
  point3 p;
  vec3 normal;
  shared_ptr<material> obj_material;
  double t;
  bool front_face;

  void set_normal(const ray &r, const vec3 &outward_normal)
  {

    // Finds normal direction based on whether we are shooting
    // inside or outside of the sphere
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = outward_normal;
    if (!front_face)
    {
      normal *= -1;
    }
  }
};

class hittable
{
public:
  virtual ~hittable() = default;

  virtual bool hit(const ray &r, interval valid_interval, hit_record &rec) const = 0;
};

#endif