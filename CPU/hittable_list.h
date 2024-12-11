#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

#include <vector>

class hittable_list : public hittable
{
public:
    std::vector<shared_ptr<hittable>> objects;

    hittable_list() {}
    hittable_list(shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }

    void add(shared_ptr<hittable> object)
    {
        objects.push_back(object);
    }

    bool hit(const ray &r, interval valid_interval, hit_record &rec) const override
    {
        hit_record temp_rec;
        bool object_hit = false;
        auto min_object_dist = valid_interval.max;

        for (const auto &object : objects)
        {
            // Find the object that collides with the ray that is closest to the camera
            if (object->hit(r, interval(valid_interval.min, min_object_dist), temp_rec))
            {
                object_hit = true;
                min_object_dist = temp_rec.t;
                rec = temp_rec;
            }
        }

        return object_hit;
    }
};

#endif