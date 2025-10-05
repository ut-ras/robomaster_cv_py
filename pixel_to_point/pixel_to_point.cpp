// pixel_to_point.cpp
// Build: g++ -std=c++17 pixel_to_point.cpp -o ptp -lrealsense2

#include <librealsense2/rs.hpp>
#include <vector>
#include <cmath>

// Replace with your actual BoundingBox header
class BoundingBox {
public:
    float get_x_center() const;  // pixel x (float)
    float get_y_center() const;  // pixel y (float)
    float get_depth()     const; // depth in meters
    void  set_x_coord(float);    // meters
    void  set_y_coord(float);    // meters
    void  set_z_coord(float);    // meters
};

// Converts (pixel x,y) + depth (m) -> point (m), following your convention:
// return (-y, -x, z) to match the comment:
//   x axis: horizontal, y axis: vertical, z axis: forward
inline std::array<float,3> convert_pixel_and_depth_to_point(
    float x_px, float y_px, float depth_m, const rs2_intrinsics& intr_in)
{
    rs2_intrinsics intr = intr_in;            // make a copy we can tweak
    intr.model = RS2_DISTORTION_NONE;         // match Python forcing 'none' model

    float pixel[2] = { x_px, y_px };
    float point[3] = { 0.f, 0.f, 0.f };       // point in camera coords (meters)
    rs2_deproject_pixel_to_point(point, &intr, pixel, depth_m);

    // Python returns: -result[1], -result[0], result[2]
    return { -point[1], -point[0], point[2] };
}

// For each bounding box, compute its 3D point from center pixel + depth and store it
inline void set_point_coords(std::vector<BoundingBox>& boxes, const rs2_intrinsics& intrinsics)
{
    for (auto& b : boxes) {
        const float x_px = b.get_x_center();
        const float y_px = b.get_y_center();
        const float z_m  = b.get_depth();     // already in meters

        auto xyz = convert_pixel_and_depth_to_point(x_px, y_px, z_m, intrinsics);
        b.set_x_coord(xyz[0]);
        b.set_y_coord(xyz[1]);
        b.set_z_coord(xyz[2]);
    }
}

