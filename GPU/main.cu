#include "utilities.h"
#include <time.h>
#include <curand_kernel.h>

#include "sphere.h"
#include "camera.h"
#include "material.h"

#define RND (curand_uniform(&rand_state))

const int IMAGE_WIDTH = 1200;
const int IMAGE_HEIGHT = 800;
const int NUM_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT;
const int NUM_SAMPLES_PER_PIXEL = 22;
const int RAY_BOUNCE_LIMIT = 50;
const int BLOCK_SIZE_X = 16;
const int BLOCK_SIZE_Y = 2;
const int OBJECT_GENERATION_SCALE_FACTOR = 7;
const int OBJECTS_IN_SCENE = (OBJECT_GENERATION_SCALE_FACTOR * 2) * (OBJECT_GENERATION_SCALE_FACTOR * 2) + 4;

#define checkCudaForErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 ray_color(const ray &r, hittable **shared_mem, curandState *local_rand_state)
{
    ray current_ray = r;
    vec3 cumulative_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < RAY_BOUNCE_LIMIT; i++)
    {
        hit_record rec;

        // Find the object that collides with the ray that is closest to the camera
        hit_record temp_rec;
        bool object_hit = false;
        float min_object_dist = FLT_MAX;

        for (int i = 0; i < OBJECTS_IN_SCENE; i++)
        {
            if (shared_mem[i]->hit(current_ray, 0.001f, min_object_dist, temp_rec))
            {
                object_hit = true;
                min_object_dist = temp_rec.t;
                rec = temp_rec;
            }
        }

        // If collision, reflect off of sphere
        if (object_hit)
        {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(current_ray, rec, attenuation, scattered, local_rand_state))
            {
                current_ray = scattered;
                cumulative_attenuation *= attenuation;
            }
            else
            {
                return color(0, 0, 0);
            }
        }
        // If no spheres are hit, collide with sky
        else
        {
            vec3 unit_direction = unit_vector(current_ray.direction());
            float a = 0.5f * (unit_direction.y() + 1.0f);
            vec3 calculated_color = (1.0f - a) * vec3(1.0, 1.0, 1.0) + a * vec3(0.4, 0.6, 1.0);
            return cumulative_attenuation * calculated_color;
        }
    }
    return color(0, 0, 0);
}

__global__ void render(vec3 *color_buffer, int max_x, int max_y, int num_samples, camera **cam, hittable **object_list, curandState *rand_state_arr)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= max_x) || (y >= max_y))
        return;
    int pixel_index = y * max_x + x;

    // initialize shared memory
    extern __shared__ hittable *shared_mem[];
    for (int32_t i = threadIdx.x; i < OBJECTS_IN_SCENE; i += blockDim.x)
    {
        shared_mem[i] = object_list[i];
    }

    curandState rand_state = rand_state_arr[pixel_index];

    // Calculate pixel color as the average of the random distribution around it
    vec3 generated_color(0, 0, 0);
    for (int s = 0; s < num_samples; s++)
    {
        float u = float(x + curand_uniform(&rand_state)) / float(max_x);
        float v = float(y + curand_uniform(&rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        generated_color += ray_color(r, shared_mem, &rand_state);
    }
    rand_state_arr[pixel_index] = rand_state;

    generated_color /= float(num_samples);
    color_buffer[pixel_index] = generated_color;
}

__global__ void rand_init(int max_x, int max_y, curandState *rand_state)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= max_x) || (y >= max_y))
        return;
    int pixel_index = y * max_x + x;
    curand_init(1099 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void create_world(hittable **object_list, camera **camera_obj, int nx, int ny, curandState *rand_state_arr)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState rand_state = *rand_state_arr;

        // Create ground
        auto ground_material = new lambertian(vec3(0.67, 0.57, 0.34));
        object_list[0] = new sphere(vec3(0, -1000.0, -1), 1000, ground_material);

        int i = 1;
        for (int a = -OBJECT_GENERATION_SCALE_FACTOR; a < OBJECT_GENERATION_SCALE_FACTOR; a++)
        {
            for (int b = -OBJECT_GENERATION_SCALE_FACTOR; b < OBJECT_GENERATION_SCALE_FACTOR; b++)
            {
                // Create a random material
                float choose_mat = curand_uniform(&rand_state);
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.80f)
                {
                    // Diffuse Material
                    object_list[i++] = new sphere(center, 0.2,
                                                  new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f)
                {
                    // Metal Material
                    auto albedo = 1.0f + RND;
                    auto fuzz = 0.5f * RND;

                    object_list[i++] = new sphere(center, 0.2,
                                                  new metal(vec3(0.5f * albedo, 0.5f * albedo, 0.5f * albedo), fuzz));
                }
                else
                {
                    // Glass Material
                    object_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }

        // Creates three giant spheres
        auto glass_material = new dielectric(1.5);
        object_list[i++] = new sphere(vec3(-2, 1, 3), 1.0, glass_material);
        auto diffuse_material = new lambertian(vec3(0.2, 0.2, 0.8));
        object_list[i++] = new sphere(vec3(0, 1, -2), 1.0, diffuse_material);
        auto metal_material = new metal(vec3(0.7, 0.6, 0.5), 0.7);
        object_list[i++] = new sphere(vec3(3, 1, 0), 1.0, metal_material);

        // Replaces random state
        *rand_state_arr = rand_state;

        // Creates camera
        const vec3 camera_pos(2, 2, 13);
        const vec3 lookat(0, 0, 0);
        const float field_of_view = 23.0;

        *camera_obj = new camera(camera_pos,
                                 lookat,
                                 vec3(0, 1, 0),
                                 field_of_view,
                                 float(nx) / float(ny));
    }
}

__global__ void free_world(hittable **object_list, camera **camera_obj)
{
    for (int i = 0; i < OBJECTS_IN_SCENE; i++)
    {
        delete ((sphere *)object_list[i])->mat_ptr;
        delete object_list[i];
    }
    delete *camera_obj;
}

int main()
{
    std::cerr << IMAGE_WIDTH << "x" << IMAGE_HEIGHT << " image with " << NUM_SAMPLES_PER_PIXEL << " samples per pixel ";
    std::cerr << "seperated into " << BLOCK_SIZE_X << "x" << BLOCK_SIZE_Y << " blocks.\n";

    // Allocate shared buffer on CPU/GPU to store buffers
    vec3 *color_buffer;
    checkCudaForErrors(cudaMallocManaged((void **)&color_buffer, NUM_PIXELS * sizeof(vec3)));

    // Create GPU random state for world creation
    curandState *random_world_init_state;
    checkCudaForErrors(cudaMalloc((void **)&random_world_init_state, 1 * sizeof(curandState)));
    rand_init<<<1, 1>>>(IMAGE_WIDTH, IMAGE_HEIGHT, random_world_init_state);
    checkCudaForErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    // Intitialize world on GPU
    hittable **object_list;
    checkCudaForErrors(cudaMalloc((void **)&object_list, OBJECTS_IN_SCENE * sizeof(hittable *)));
    camera **camera_obj;
    checkCudaForErrors(cudaMalloc((void **)&camera_obj, sizeof(camera *)));
    create_world<<<1, 1>>>(object_list, camera_obj, IMAGE_WIDTH, IMAGE_HEIGHT, random_world_init_state);
    checkCudaForErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    // Create per thread GPU random state for render calcualtions
    int num_blocks_x = IMAGE_WIDTH / BLOCK_SIZE_X;
    num_blocks_x += IMAGE_WIDTH % BLOCK_SIZE_X != 0;
    int num_blocks_y = IMAGE_WIDTH / BLOCK_SIZE_Y;
    num_blocks_y += IMAGE_WIDTH % BLOCK_SIZE_Y != 0;
    dim3 num_blocks(num_blocks_x, num_blocks_y);
    dim3 threads_per_block(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    curandState *random_state_arr;
    checkCudaForErrors(cudaMalloc((void **)&random_state_arr, NUM_PIXELS * sizeof(curandState)));
    rand_init<<<num_blocks, threads_per_block>>>(IMAGE_WIDTH, IMAGE_HEIGHT, random_state_arr);
    checkCudaForErrors(cudaGetLastError());
    cudaDeviceSynchronize();

    // Perform render
    clock_t start, stop;
    start = clock();
    uint32_t shared_mem_size = sizeof(hittable *) * (OBJECTS_IN_SCENE + 10);
    render<<<num_blocks, threads_per_block, shared_mem_size>>>(color_buffer, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_SAMPLES_PER_PIXEL, camera_obj, object_list, random_state_arr);
    checkCudaForErrors(cudaGetLastError());
    cudaDeviceSynchronize();
    stop = clock();

    // Calculate benchmarks
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Rendering Time: " << timer_seconds << " sec \n";
    std::cerr << "Frames Per Second: " << 1.0 / timer_seconds << " fps \n";
    std::cerr << "Pixels Per Second: " << 1.0 / (timer_seconds / NUM_PIXELS) << " fps \n";

    // Compile results into .ppm image file
    std::cout << "P3\n"
              << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
    for (int j = IMAGE_HEIGHT - 1; j >= 0; j--)
    {
        for (int i = 0; i < IMAGE_WIDTH; i++)
        {
            size_t pixel_index = j * IMAGE_WIDTH + i;
            write_color(std::cout, color_buffer[pixel_index]);
        }
    }

    cudaDeviceSynchronize();
    free_world<<<1, 1>>>(object_list, camera_obj);
    checkCudaForErrors(cudaGetLastError());
    checkCudaForErrors(cudaFree(camera_obj));
    checkCudaForErrors(cudaFree(object_list));
    checkCudaForErrors(cudaFree(random_state_arr));
    checkCudaForErrors(cudaFree(random_world_init_state));
    checkCudaForErrors(cudaFree(color_buffer));
}