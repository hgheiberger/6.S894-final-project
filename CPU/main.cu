#include "utilities.h"
#include <time.h>

#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

const int IMAGE_WIDTH = 1200;
const double IMAGE_ASPECT_RATIO = 16.0 / 9.0;
const int NUM_SAMPLES_PER_PIXEL = 22;
const int RAY_BOUNCE_LIMIT = 50;
const int OBJECT_GENERATION_SCALE_FACTOR = 7;
const point3 CAMERA_INITAL_POS = point3(2, 2, 13);
const point3 CAMERA_LOOK_POS = point3(0, 0, 0);
const int CAMERA_VERTICAL_FOV = 20;

int main()
{

    // World
    hittable_list world;

    // Create ground
    auto ground_material = make_shared<lambertian>(color(0.67, 0.57, 0.34));
    world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -OBJECT_GENERATION_SCALE_FACTOR; a < OBJECT_GENERATION_SCALE_FACTOR; a++)
    {
        for (int b = -OBJECT_GENERATION_SCALE_FACTOR; b < OBJECT_GENERATION_SCALE_FACTOR; b++)
        {
            // Randomly choose location and material
            auto material_prob = random_double();
            point3 center = point3(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            // Create sphere
            if ((center - point3(4, 0.2, 0)).length() > 0.9)
            {
                shared_ptr<material> sphere_material;
                if (material_prob < 0.8)
                {
                    // Diffuse Material
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (material_prob < 0.95)
                {
                    // Metal Material
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
                else
                {
                    // Glass Material
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    // Generate centered giant spheres
    auto glass_material = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(-2, 1, 3), 1.0, glass_material));
    auto diffuse_material = make_shared<lambertian>(color(0.2, 0.2, 0.8));
    world.add(make_shared<sphere>(point3(0, 1, -2), 1.0, diffuse_material));
    auto metal_material = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(3, 1, 0), 1.0, metal_material));

    // Setup camera
    camera cam;
    cam.aspect_ratio = IMAGE_ASPECT_RATIO;
    cam.image_width = IMAGE_WIDTH;
    cam.samples_per_pixel = NUM_SAMPLES_PER_PIXEL;
    cam.max_ray_bounces = RAY_BOUNCE_LIMIT;
    cam.vertical_fov = CAMERA_VERTICAL_FOV;
    cam.camera_pos = CAMERA_INITAL_POS;
    cam.lookat = CAMERA_LOOK_POS;
    cam.vup = vec3(0, 1, 0);

    clock_t start, stop;
    start = clock();
    cam.render(world);
    stop = clock();

    // Calculate benchmarks
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering Time: " << timer_seconds << " sec \n";
    std::cerr << "Frames Per Second: " << 1.0 / timer_seconds << " fps \n";
}