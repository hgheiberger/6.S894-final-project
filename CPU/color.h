#ifndef COLOR_H
#define COLOR_H

#include "interval.h"
#include "vec3.h"

using color = vec3;

inline double convert_linear_color_to_gamma(double linear_color)
{
    if (linear_color > 0)
        return std::sqrt(linear_color);

    return 0;
}

void write_color(std::ostream &out, const color &pixel_color)
{
    // linear to gamma transform
    auto r = convert_linear_color_to_gamma(pixel_color.x());
    auto g = convert_linear_color_to_gamma(pixel_color.y());
    auto b = convert_linear_color_to_gamma(pixel_color.z());

    // Convert [0,1] range to [0, 255] range
    static const interval intensity(0.000, 0.999);
    int scaled_r = int(256 * intensity.clamp(r));
    int scaled_g = int(256 * intensity.clamp(g));
    int scaled_b = int(256 * intensity.clamp(b));

    // Output ppm format to inputted ostream
    out << scaled_r << ' ' << scaled_g << ' ' << scaled_b << '\n';
}

#endif