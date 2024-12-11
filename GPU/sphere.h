#ifndef SPHEREH
#define SPHEREH

#include "hittable.h"

class sphere: public hittable  {
    public:
        __device__ sphere(const point3 center, float radius, material *material) : center(center), radius(radius), mat_ptr(material)  {};
        
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const{
          vec3 oc = r.origin() - center;
          float a = r.direction().length_squared();
          float h = dot(oc, r.direction());
          float c = oc.length_squared() - radius*radius;
          
          float discriminant = h*h - a*c;
          if (discriminant < 0){
              return false;
          }

          auto sqrtd = std::sqrt(discriminant);

          float root = (-h - sqrtd)/a;
          if (!(root < t_max && root > t_min)) {
              root = (-h + sqrtd) / a;
              if (!(root < t_max && root > t_min)) {
                return false;
              }
          }
          
          rec.t = root;
          rec.p = r.point_at_parameter(rec.t);
          rec.normal = (rec.p - center) / radius;
          rec.mat_ptr = mat_ptr;

          return true;
        }
      material *mat_ptr;
      
      private:
        point3 center;
        float radius;
};


#endif