// BoundingBox.hpp
#pragma once
#include <utility>   // std::pair
#include <ctime>     // std::time_t, std::tm, std::localtime
#include <iomanip>   // std::put_time
#include <iostream>  // std::ostream

class BoundingBox {
private:
    float x1_{0.f}, x2_{0.f};
    float y1_{0.f}, y2_{0.f};
    float x_center_{0.f}, y_center_{0.f};
    float depth_value_{0.f};
    float x_coord_{0.f};  // meters
    float y_coord_{0.f};  // meters
    float z_coord_{0.f};  // meters
    float height_{0.f}, width_{0.f};
    std::tm time_{};      // local time
    bool has_time_{false};

public:
    BoundingBox() = default;

    // ---- Setters ----
    void set_x_value(float x1, float x2) {
        x1_ = x1; x2_ = x2;
    }

    void set_y_value(float y1, float y2) {
        y1_ = y1; y2_ = y2;
    }

    void set_depth(float depth)            { depth_value_ = depth; }
    void set_x_coord(float x_coord)        { x_coord_ = x_coord; }
    void set_y_coord(float y_coord)        { y_coord_ = y_coord; }
    void set_z_coord(float z_coord)        { z_coord_ = z_coord; }

    void calculate_x_center()              { x_center_ = (x1_ + x2_) * 0.5f; }
    void calculate_y_center()              { y_center_ = (y1_ + y2_) * 0.5f; }
    void calculate_height()                { height_   = (y2_ - y1_); }
    void calculate_width()                 { width_    = (x2_ - x1_); }

    void set_time() {
        std::time_t t = std::time(nullptr);
        time_ = *std::localtime(&t); // same semantics as Python localtime()
        has_time_ = true;
    }

    // ---- Getters ----
    std::pair<float,float> get_x_value() const { return {x1_, x2_}; }
    std::pair<float,float> get_y_value() const { return {y1_, y2_}; }

    float get_x_center() {
        calculate_x_center();  // Python recomputes on access
        return x_center_;
    }

    float get_y_center() {
        calculate_y_center();  // Python recomputes on access
        return y_center_;
    }

    float get_depth()    const { return depth_value_; }
    float get_x_coord()  const { return x_coord_; }
    float get_y_coord()  const { return y_coord_; }
    float get_z_coord()  const { return z_coord_; }
    float get_height()   const { return height_; }
    float get_width()    const { return width_;  }

    const std::tm& get_time() const { return time_; }
    bool has_time() const { return has_time_; }

    void print_time(std::ostream& os = std::cout) const {
        if (!has_time_) { os << "<time not set>\n"; return; }
        // Format similar to Python's struct_time default-ish look
        os << std::put_time(&time_, "%a %b %d %H:%M:%S %Y") << '\n';
    }
};

