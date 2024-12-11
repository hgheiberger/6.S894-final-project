#ifndef INTERVAL_H
#define INTERVAL_H

class interval
{
public:
    double min, max;

    interval(double min, double max) : min(min), max(max) {}

    double size() const
    {
        return max - min;
    }

    double clamp(double x) const
    {
        if (x < min)
            return min;
        if (x > max)
            return max;
        return x;
    }

    bool contains(double x) const
    {
        return min <= x && x <= max;
    }

    bool surrounds(double x) const
    {
        return min < x && x < max;
    }

    static const interval empty, universe;
};

#endif