#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "material.h"

class camera
{
public:
    int image_width = 1200;
    double aspect_ratio = (16.0 / 9.0);
    int samples_per_pixel = 22;
    int max_ray_bounces = 50;
    point3 camera_pos = point3(13, 2, 3);
    point3 lookat = point3(0, 0, 0);
    vec3 vup = vec3(0, 1, 0); // Camera-relative "up" direction
    double vertical_fov = 20;

    void render(const hittable &world)
    {
        // Initialize scene information
        initialize();

        // Compile results into .ppm image file
        std::cout << "P3\n"
                  << image_width << ' ' << image_height << "\n255\n";
        for (int j = 0; j < image_height; j++)
        {
            for (int i = 0; i < image_width; i++)
            {
                color pixel_color(0, 0, 0);
                // Get color of pixel through antialiasing sampling
                for (int sample = 0; sample < samples_per_pixel; sample++)
                {
                    ray r = get_ray_around_point(i, j);
                    pixel_color += calculate_ray_color(r, max_ray_bounces, world);
                }
                write_color(std::cout, pixel_color * (1.0 / samples_per_pixel));
            }
        }
    }

private:
    int image_height;
    point3 pixel_corner_loc; // (0,0)
    vec3 pixel_delta_u;
    vec3 pixel_delta_v;
    vec3 u_basis;
    vec3 v_basis;
    vec3 w_basis;

    void initialize()
    {
        image_height = int(image_width / aspect_ratio);
        image_height = max(1, image_height);

        // Calculate camera characteristics
        auto focal_length = (camera_pos - lookat).length();
        auto theta = degrees_to_radians(vertical_fov);
        auto h = std::tan(theta / 2);
        auto viewport_height = 2 * h * focal_length;
        auto viewport_width = viewport_height * (double(image_width) / image_height);

        // u,v,w unit basis vectors of the camera coordinate system.
        w_basis = unit_vector(camera_pos - lookat);
        u_basis = unit_vector(cross(vup, w_basis));
        v_basis = cross(w_basis, u_basis);

        // Create viewport between camera and 3d world
        vec3 viewport_u = viewport_width * u_basis;
        vec3 viewport_v = viewport_height * -v_basis;

        // Pixel to pixel delta
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        auto viewport_upper_left = camera_pos - (focal_length * w_basis) - viewport_u / 2 - viewport_v / 2;
        pixel_corner_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
    }

    // Gets ray from camera origin directed at random point around i, j
    ray get_ray_around_point(int i, int j) const
    {
        auto random_offset = vec3(random_double() - 0.5, random_double() - 0.5, 0);
        auto pixel_sample = pixel_corner_loc + ((i + random_offset.x()) * pixel_delta_u) + ((j + random_offset.y()) * pixel_delta_v);
        return ray(camera_pos, pixel_sample - camera_pos);
    }

    color calculate_ray_color(const ray &r, int remaining_bounces, const hittable &world) const
    {
        // Light bounce limit reached
        if (remaining_bounces <= 0)
            return color(0, 0, 0);

        // If collision with object
        hit_record rec;
        if (world.hit(r, interval(0.001, infinity), rec))
        {
            ray scattered_ray;
            color attenuation;

            // If the ray reflects (based on the material)
            if (rec.obj_material->scatter(r, rec, attenuation, scattered_ray))
                return attenuation * calculate_ray_color(scattered_ray, remaining_bounces - 1, world);

            // If the ray is absorbed
            return color(0, 0, 0);
        }

        // Collide with sky
        vec3 unit_direction = unit_vector(r.direction());
        auto a = 0.5 * (unit_direction.y() + 1.0);
        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.4, 0.6, 1.0);
    }
};

#endif