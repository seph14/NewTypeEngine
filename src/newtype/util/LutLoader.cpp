#include "newtype/util/LutLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>

namespace newtype::util {

LutData loadCubeLut(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Failed to open LUT file: " + path.string());

    LutData result;
    std::string line;

    while (std::getline(file, line)) {
        // Strip leading whitespace
        auto pos = line.find_first_not_of(" \t\r\n");
        if (pos == std::string::npos) continue;
        line = line.substr(pos);

        // Skip comments
        if (line[0] == '#') continue;

        // Parse header keywords
        if (line.rfind("LUT_3D_SIZE", 0) == 0) {
            std::istringstream iss(line.substr(11));
            iss >> result.size;
            if (result.size == 0)
                throw std::runtime_error("Invalid LUT_3D_SIZE in: " + path.string());
            continue;
        }
        if (line.rfind("TITLE", 0) == 0) {
            // TITLE "name" — extract quoted string
            auto q1 = line.find('"');
            auto q2 = line.rfind('"');
            if (q1 != std::string::npos && q2 > q1)
                result.title = line.substr(q1 + 1, q2 - q1 - 1);
            continue;
        }

        // Skip other header keywords (DOMAIN_MIN, DOMAIN_MAX, LUT_1D_SIZE, etc.)
        if (std::isalpha(static_cast<unsigned char>(line[0])))
            continue;

        // Data line: R G B
        float r, g, b;
        std::istringstream iss(line);
        if (iss >> r >> g >> b) {
            result.data.push_back(r);
            result.data.push_back(g);
            result.data.push_back(b);
            result.data.push_back(1.f); // directly align 4flts RGBA in loader
        }
    }

    // Validate
    uint32_t expected = result.size * result.size * result.size * 3u;
    if (result.data.size() * 3 / 4 != expected)
        throw std::runtime_error(
            "LUT data size mismatch in " + path.string() +
            ": expected " + std::to_string(expected) +
            " floats, got " + std::to_string(result.data.size() * 3 / 4));

    if (result.size == 0)
        throw std::runtime_error("No LUT_3D_SIZE found in: " + path.string());

    return result;
}

} // namespace newtype::core
