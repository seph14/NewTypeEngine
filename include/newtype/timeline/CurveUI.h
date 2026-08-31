#pragma once

#include "cinder/CinderImGui.h"

namespace ImGui {

template <int steps>
void bezier_table(ImVec2 P[], ImVec2 results[]) {
    static float C[(steps + 1) * 4], *K = nullptr;
    if (!K) {
        K = C;
        for (unsigned step = 0; step <= steps; ++step) {
            float t = (float)step / (float)steps;
            C[step * 4 + 0] = (1.f - t) * (1.f - t) * (1.f - t);
            C[step * 4 + 1] = 3.f * (1.f - t) * (1.f - t) * t;
            C[step * 4 + 2] = 3.f * (1.f - t) * t * t;
            C[step * 4 + 3] = t * t * t;
        }
    }
    for (unsigned step = 0; step <= steps; ++step) {
        ImVec2 point = {
            K[step * 4 + 0] * P[0].x + K[step * 4 + 1] * P[1].x +
            K[step * 4 + 2] * P[2].x + K[step * 4 + 3] * P[3].x,
            K[step * 4 + 0] * P[0].y + K[step * 4 + 1] * P[1].y +
            K[step * 4 + 2] * P[2].y + K[step * 4 + 3] * P[3].y};
        results[step] = point;
    }
}

float BezierValue(float dt01, float P[]);
int  Bezier(const char* label, float P[5]);

} // namespace ImGui
