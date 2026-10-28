#pragma once
#include <string_view>
#include <vector>

namespace cv {
// Owns normalized RGB CHW pixels. data must contain width * height * 3 values.
struct CvImage{
    std::vector<float> data;
    int width = 0;
    int height = 0;
};

struct Color{
    float r;
    float g;
    float b;
};

struct Point {
    int x;
    int y;
};

struct Dim {
    int width;
    int height;
};

struct Rect {
    int x;
    int y;
    int width;
    int height;
};

enum class Interpolation {
    Nearest, //STBIR_FILTER_POINT_SAMPLE
    Bilinear, //STBIR_FILTER_TRIANGLE
    Bicubic //STBIR_FILTER_CATMULLROM
};

namespace colors {
inline constexpr Color white  {1.0F, 1.0F, 1.0F};
inline constexpr Color black  {0.0F, 0.0F, 0.0F};
inline constexpr Color red    {1.0F, 0.0F, 0.0F};
inline constexpr Color green  {0.0F, 1.0F, 0.0F};
inline constexpr Color blue   {0.0F, 0.0F, 1.0F};
inline constexpr Color yellow {1.0F, 1.0F, 0.0F};
}

void setPixel(CvImage& image, Point position, Color color);
void line(CvImage& image, Point from, Point to, Color color, int thickness=1);
void rectangle(CvImage& image, Rect rect, Color color, int thickness=1);
void fillRect(CvImage& image, Rect rect, Color color);
Dim getTextSize(std::string_view text, int scale=1);
void putText(CvImage& image, std::string_view text, Point position, Color color, int scale=1);
void resize(const CvImage& src, CvImage& dst, Dim outputSize, Interpolation interpolation=Interpolation::Bilinear);
void drawBbox(CvImage& image, const std::vector<float>& output, int class_count, int thickness=2);
}
