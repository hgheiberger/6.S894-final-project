#ifndef CAMERAH
#define CAMERAH

class camera {
    public:
        __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect_ratio) {
            float half_height = tan(degrees_to_radians(vfov)/2);
            float half_width = half_height * aspect_ratio;
            camera_coord = lookfrom;

            // u,v,w unit basis vectors of camera coordinates
            vec3 w_basis = unit_vector(lookfrom - lookat);
            vec3 u_basis = unit_vector(cross(vup, w_basis));
            vec3 v_basis  = cross(w_basis, u_basis);

            viewpoint_corner = camera_coord - half_width*u_basis - half_height*v_basis - w_basis;
            horizontal = 2*half_width*u_basis;
            vertical = 2*half_height*v_basis;
        }

        __device__ ray get_ray(float u, float v) { return ray(camera_coord, viewpoint_corner + u*horizontal + v*vertical - camera_coord); }

        vec3 camera_coord;
        vec3 viewpoint_corner;
        vec3 horizontal;
        vec3 vertical;
};

#endif