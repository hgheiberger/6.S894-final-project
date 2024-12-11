#ifndef UTILITIES_H
#define UTILITIES_H

#include <cmath>
#include <float.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

// Constants
const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility functions
__device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Common Headers
#include "ray.h"
#include "vec3.h"
#include "color.h"
#include "interval.h"

#endif