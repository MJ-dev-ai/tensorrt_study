#include "cv/cv.hpp"
#include "stb_easy_font.h"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <cstdio>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
#include <utility>

namespace cv {
namespace {
// Check dimensions before multiplication or allocation.
size_t imageSize(int width, int height) {
    if (width <= 0 || height <= 0) throw std::invalid_argument("Image dimensions must be positive");
    const size_t w = static_cast<size_t>(width);
    const size_t h = static_cast<size_t>(height);
    if (w > std::vector<float>().max_size() / 3 / h) throw std::length_error("Image is too large");
    return w * h * 3;
}

// Public image fields must describe a complete RGB CHW buffer.
void validateImage(const CvImage& image) {
    if (image.data.size() < imageSize(image.width, image.height)) {
        throw std::invalid_argument("Image buffer is smaller than its dimensions");
    }
}

// stb accepts printable ASCII and newline; replace other bytes consistently.
std::string fontText(std::string_view text) {
    std::string result{text};
    for (char& ch : result) {
        const unsigned char value = static_cast<unsigned char>(ch);
        if (ch != '\n' && (value < 32 || value > 126)) ch = '?';
    }
    return result;
}
} // namespace

void setPixel(CvImage& image, Point position, Color color) {
    validateImage(image);
    if (position.x < 0 || position.x >= image.width ||
        position.y < 0 || position.y >= image.height) {
            return;
        }
    const size_t index = static_cast<size_t>(position.y) * image.width + position.x;
    const size_t plane_size = static_cast<size_t>(image.width) * image.height;
    image.data[index] = color.r;
    image.data[plane_size + index] = color.g;
    image.data[2 * plane_size + index] = color.b;
}

void line(CvImage& image, Point from, Point to, Color color, int thickness) {
    validateImage(image);
    if (thickness <= 0) { return; }
    int x = from.x;
    int y = from.y;

    const int dx = std::abs(to.x - from.x);
    const int dy = -std::abs(to.y - from.y);

    const int step_x = from.x < to.x ? 1 : -1;
    const int step_y = from.y < to.y ? 1 : -1;

    int error = dx + dy;

    const int before = (thickness - 1) / 2;
    const int after = thickness / 2;

    while (true) {
        const int left = std::max(0, x - before);
        const int right = std::min(image.width - 1, x + after);
        const int top = std::max(0, y - before);
        const int bottom = std::min(image.height - 1, y + after);

        for (int py = top; py <= bottom; py ++) {
            for (int px = left; px <= right; px ++) {
                setPixel(image, {px, py}, color);
            }
        }

        if (x == to.x && y == to.y) { break; }

        const int twice_error = 2 * error;
        if (twice_error >= dy) {
            error += dy;
            x += step_x;
        }
        if (twice_error <= dx) {
            error += dx;
            y += step_y;
        }
    }
}

void rectangle(CvImage& image, Rect rect, Color color, int thickness) {
    validateImage(image);
    if (rect.width <= 0 || rect.height <= 0 || thickness <= 0) { return; }
    
    const int x1 = std::max(0, rect.x);
    const int y1 = std::max(0, rect.y);
    const int x2 = std::min(image.width - 1, rect.x + rect.width - 1);
    const int y2 = std::min(image.height - 1, rect.y + rect.height - 1);
    if (x1 > x2 || y1 > y2) { return; }

    line(image, {x1, y1}, {x2, y1}, color, thickness);
    line(image, {x2, y1}, {x2, y2}, color, thickness);
    line(image, {x1, y1}, {x1, y2}, color, thickness);
    line(image, {x1, y2}, {x2, y2}, color, thickness);
}

void fillRect(CvImage& image, Rect rect, Color color) {
    validateImage(image);
    if (rect.width <= 0 || rect.height <= 0) { return; }
    
    const int x1 = std::max(0, rect.x);
    const int y1 = std::max(0, rect.y);
    const int x2 = std::min(image.width - 1, rect.x + rect.width - 1);
    const int y2 = std::min(image.height - 1, rect.y + rect.height - 1);
    if (x1 > x2 || y1 > y2) { return; }

    for (int y = y1; y <= y2; y++) {
        for (int x = x1; x <= x2; x++) {
            setPixel(image, {x, y}, color);
        }
    }
}

Dim getTextSize(std::string_view text, int scale) {
    if (text.empty() || scale <= 0) { return {0, 0}; }

    std::string buffer = fontText(text);
    const int width = stb_easy_font_width(buffer.data());
    const int height = stb_easy_font_height(buffer.data());
    return {width * scale, height * scale};
}

void putText(CvImage& image, std::string_view text, Point position, Color color, int scale) {
    validateImage(image);
    if (text.empty() || scale <= 0) { return; }

    struct Vertex {
        float x, y, z;
        unsigned char color[4];
    };
    static_assert(sizeof(Vertex) == 16);

    if (text.size() > static_cast<size_t>(std::numeric_limits<int>::max()) / (32 * sizeof(Vertex))) {
        throw std::length_error("Text is too long");
    }
    std::vector<Vertex> vertices(text.size() * 32);
    std::string buffer = fontText(text);

    const int quad_count = stb_easy_font_print(
        0.0F, 0.0F, buffer.data(), nullptr, vertices.data(), static_cast<int>(vertices.size() * sizeof(Vertex))
    );

    for (int q = 0; q < quad_count; q++) {
        const Vertex* quad = vertices.data() + q * 4;

        float left = quad[0].x;
        float right = quad[0].x;
        float top = quad[0].y;
        float bottom = quad[0].y;

        for (int v = 1; v < 4; v++) {
            left = std::min(left, quad[v].x);
            right = std::max(right, quad[v].x);
            top = std::min(top, quad[v].y);
            bottom = std::max(bottom, quad[v].y);
        }

        const int x1 = position.x + static_cast<int>(std::floor(left * scale));
        const int y1 = position.y + static_cast<int>(std::floor(top * scale));
        const int x2 = position.x + static_cast<int>(std::floor(right * scale));
        const int y2 = position.y + static_cast<int>(std::floor(bottom * scale));

        fillRect(image, {x1, y1, x2 - x1, y2 - y1}, color);
    }
}

// Validate and dispatch; preserve the input when resizing an image in place.
void resize(const CvImage& src, CvImage& dst, Dim outputSize, Interpolation interpolation) {
    validateImage(src);
    const size_t output_count = imageSize(outputSize.width, outputSize.height);
    stbir_filter filter;
    switch (interpolation) {
        case Interpolation::Nearest:
            filter = STBIR_FILTER_POINT_SAMPLE;
            break;
        case Interpolation::Bilinear:
            filter = STBIR_FILTER_TRIANGLE;
            break;
        case Interpolation::Bicubic:
            filter = STBIR_FILTER_CATMULLROM;
            break;
        default:
            throw std::invalid_argument("Unsupported interpolation");
    }

    CvImage result{std::vector<float>(output_count), outputSize.width, outputSize.height};
    const size_t src_plane = static_cast<size_t>(src.width) * src.height;
    const size_t dst_plane = static_cast<size_t>(outputSize.width) * outputSize.height;

    for (int c = 0; c < 3; c++) {
        const float* input = src.data.data() + c * src_plane;
        float* output = result.data.data() + c * dst_plane;

        STBIR_RESIZE config{};
        stbir_resize_init(
            &config, input, src.width, src.height, 0,
            output, result.width, result.height, 0,
            STBIR_1CHANNEL, STBIR_TYPE_FLOAT
        );

        if (!stbir_set_filters(&config, filter, filter)) throw std::runtime_error("Failed to configure resize filter");
        if (!stbir_resize_extended(&config)) throw std::runtime_error("Failed to resize image");
    }

    dst = std::move(result);
}

// Decode one YOLO xywh/score output, apply NMS, and draw class_id:score labels.
void drawBbox(CvImage& image, const std::vector<float>& output, int class_count, int thickness) {
    validateImage(image);
    if (class_count <= 0) { throw std::invalid_argument("Invalid class count"); }
    const size_t row_count = 4 + static_cast<size_t>(class_count);
    if (output.size() % row_count != 0) { throw std::invalid_argument("Invalid output size"); }
    const size_t candidate_count = output.size() / row_count;
    if (thickness <= 0) return;

    struct Box {
        float x1, y1, x2, y2;
        float score;
        int class_id;
    };

    constexpr float CONF_THRESHOLD = 0.25F;
    constexpr float IOU_THRESHOLD = 0.45F;
    std::vector<Box> boxes;

    for (size_t row = 0; row < candidate_count; row++) {
        float score = 0.0F;
        int class_id = -1;

        for (int c = 0; c < class_count; c++) {
            const float value = output[(4 + static_cast<size_t>(c)) * candidate_count + row];

            if (std::isfinite(value) && value > score) {
                score = value;
                class_id = c;
            }
        }
        if (score < CONF_THRESHOLD) continue;

        const float cx = output[row];
        const float cy = output[candidate_count + row];
        const float width = output[2 * candidate_count + row];
        const float height = output[3 * candidate_count + row];

        if (!std::isfinite(cx) || !std::isfinite(cy) || !std::isfinite(width) || !std::isfinite(height) || width <= 0 || height <= 0) continue;

        const Box box{cx - width * 0.5F, cy - height * 0.5F, cx + width * 0.5F, cy + height * 0.5F, score, class_id};
        // Guard conversion and Rect arithmetic without changing the clipping policy.
        const double left = std::floor(static_cast<double>(box.x1));
        const double top = std::floor(static_cast<double>(box.y1));
        const double right = std::ceil(static_cast<double>(box.x2));
        const double bottom = std::ceil(static_cast<double>(box.y2));
        const double min_int = std::numeric_limits<int>::min();
        const double max_int = std::numeric_limits<int>::max();
        if (!std::isfinite(left) || !std::isfinite(top) || !std::isfinite(right) || !std::isfinite(bottom) ||
            left < min_int || top < min_int || right > max_int || bottom > max_int ||
            right <= left || bottom <= top || right - left > max_int || bottom - top > max_int) continue;
        if (right <= 0 || bottom <= 0 || left >= image.width || top >= image.height) continue;
        boxes.push_back(box);
    }

    std::sort(boxes.begin(), boxes.end(), [](const Box&a, const Box& b) { return a.score > b.score; });

    std::vector<Box> selected;

    for (const Box& box : boxes) {
        bool suppressed = false;

        for (const Box& kept : selected) {
            const float overlap_width = std::max(0.0F, std::min(box.x2, kept.x2) - std::max(box.x1, kept.x1));
            const float overlap_height = std::max(0.0F, std::min(box.y2, kept.y2) - std::max(box.y1, kept.y1));

            const double intersection = static_cast<double>(overlap_height) * overlap_width;
            const double box_area = static_cast<double>(box.x2 - box.x1) * (box.y2 - box.y1);
            const double kept_area = static_cast<double>(kept.x2 - kept.x1) * (kept.y2 - kept.y1);
            const double union_area = box_area + kept_area - intersection;

            const double iou = union_area > 0.0 ? intersection / union_area : 0.0;
            const bool same_class = box.class_id == kept.class_id;

            if ((same_class && iou > IOU_THRESHOLD) || (!same_class && iou > 0.90F)) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) selected.push_back(box);
    }

    const std::vector<Color> palette{
        colors::red,
        colors::green,
        colors::blue,
        colors::yellow,
        {0.0F, 1.0F, 1.0F}, // 청록
        {1.0F, 0.0F, 1.0F}, // 자홍
        {1.0F, 0.5F, 0.0F}, // 주황
        {0.5F, 0.0F, 1.0F}, // 보라
    };

    for (const Box& box : selected) {
        const int x1 = static_cast<int>(std::floor(box.x1));
        const int y1 = static_cast<int>(std::floor(box.y1));
        const int x2 = static_cast<int>(std::ceil(box.x2));
        const int y2 = static_cast<int>(std::ceil(box.y2));
        const Color color = palette[static_cast<size_t>(box.class_id) % palette.size()];
        rectangle(image, {x1, y1, x2 - x1, y2 - y1}, color, thickness);

        char label[64];
        std::snprintf(label, sizeof(label), "%d:%.2f", box.class_id, box.score);

        const int scale = 2;

        const Dim text_size = getTextSize(label, scale);

        const int text_x = std::clamp(x1, 0, std::max(0, image.width - text_size.width));
        const int preferred_y = y1 >= text_size.height + 2 ? y1 - text_size.height - 2 : y1 + 2;

        const int text_y = std::clamp(preferred_y, 0, std::max(0, image.height - text_size.height));

        putText(image, label, {text_x, text_y}, color, scale);
    }
}
} // namespace cv