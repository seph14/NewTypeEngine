#include "newtype/timeline/CurveUI.h"
#include "imgui/imgui_internal.h"

namespace ImGui {

float BezierValue(float dt01, float P[]) {
    enum { STEPS = 256 };
    ImVec2 Q[4] = { {0, 0}, {P[0], P[1]}, {P[2], P[3]}, {1, 1} };
    ImVec2 results[STEPS + 1];
    bezier_table<STEPS>(Q, results);
    return results[(int)((dt01 < 0 ? 0 : dt01 > 1 ? 1 : dt01) * STEPS)].y;
}

int Bezier(const char* label, float P[]) {
    enum { SMOOTHNESS = 64 };
    enum { CURVE_WIDTH = 4 };
    enum { LINE_WIDTH  = 1 };
    enum { GRAB_RADIUS = 8 };
    enum { GRAB_BORDER = 2 };
    enum { AREA_CONSTRAINED = true };
    enum { AREA_WIDTH = 0 };
    enum { AREA_HEIGHT= 150 };

    static struct {
        const char* name;
        float points[4];
    } presets[] = {
        {"Linear",      {0.000f, 0.000f, 1.000f, 1.000f}},
        {"In Sine",     {0.470f, 0.000f, 0.745f, 0.715f}},
        {"In Quad",     {0.550f, 0.085f, 0.680f, 0.530f}},
        {"In Cubic",    {0.550f, 0.055f, 0.675f, 0.190f}},
        {"In Quart",    {0.895f, 0.030f, 0.685f, 0.220f}},
        {"In Quint",    {0.755f, 0.050f, 0.855f, 0.060f}},
        {"In Expo",     {0.950f, 0.050f, 0.795f, 0.035f}},
        {"In Circ",     {0.600f, 0.040f, 0.980f, 0.335f}},
        {"In Back",     {0.600f, -0.28f, 0.735f, 0.045f}},
        {"Out Sine",    {0.390f, 0.575f, 0.565f, 1.000f}},
        {"Out Quad",    {0.250f, 0.460f, 0.450f, 0.940f}},
        {"Out Cubic",   {0.215f, 0.610f, 0.355f, 1.000f}},
        {"Out Quart",   {0.165f, 0.840f, 0.440f, 1.000f}},
        {"Out Quint",   {0.230f, 1.000f, 0.320f, 1.000f}},
        {"Out Expo",    {0.190f, 1.000f, 0.220f, 1.000f}},
        {"Out Circ",    {0.075f, 0.820f, 0.165f, 1.000f}},
        {"Out Back",    {0.175f, 0.885f, 0.320f, 1.275f}},
        {"InOut Sine",  {0.445f, 0.050f, 0.550f, 0.950f}},
        {"InOut Quad",  {0.455f, 0.030f, 0.515f, 0.955f}},
        {"InOut Cubic", {0.645f, 0.045f, 0.355f, 1.000f}},
        {"InOut Quart", {0.770f, 0.000f, 0.175f, 1.000f}},
        {"InOut Quint", {0.860f, 0.000f, 0.070f, 1.000f}},
        {"InOut Expo",  {1.000f, 0.000f, 0.000f, 1.000f}},
        {"InOut Circ",  {0.785f, 0.135f, 0.150f, 0.860f}},
        {"InOut Back",  {0.680f, -0.55f, 0.265f, 1.550f}},
    };

    bool reload = 0;
    ImGui::PushID(label);
    if (ImGui::ArrowButton("##lt", ImGuiDir_Left)) {
        if (--P[4] >= 0) reload = 1; else ++P[4];
    }
    ImGui::SameLine();

    if (ImGui::Button("Presets")) {
        ImGui::OpenPopup("!Presets");
    }

    if (ImGui::BeginPopup("!Presets")) {
        for (int i = 0; i < IM_ARRAYSIZE(presets); ++i) {
            if (i == 1 || i == 9 || i == 17)
                ImGui::Separator();
            if (ImGui::MenuItem(presets[i].name, NULL, P[4] == i)) {
                P[4] = i;
                reload = 1;
            }
        }
        ImGui::EndPopup();
    }
    ImGui::SameLine();

    if (ImGui::ArrowButton( "##rt", ImGuiDir_Right)) {
        if (++P[4] < IM_ARRAYSIZE(presets))
            reload = 1;
        else
            --P[4];
    }
    ImGui::PopID();

    if (reload) {
        memcpy(P, presets[(int)P[4]].points, sizeof(float) * 4);
    }

    ImDrawList* DrawList = GetWindowDrawList();
    const ImGuiStyle& Style = GetStyle();

    int changed = 0;
    changed = SliderFloat4("", P, 0, 2, "%.3f", 1.0f);
    int hovered = IsItemActive() || IsItemHovered();
    Dummy(ImVec2(0, 3));

    const float avail = GetContentRegionAvail().x;
    const float dim_w = (int)AREA_WIDTH > 0 ? AREA_WIDTH : avail;
    const float dim_h = AREA_HEIGHT;
    ImVec2 Canvas(dim_w, dim_h);

    ImVec2 cursor = GetCursorScreenPos();
    ImRect bb(cursor, cursor + Canvas);
    //ItemSize(bb);
    
    // Use ImHashStr for ID generation instead of Window->GetID
    ImGuiID id = ImHashStr(label);
    
    // Create invisible button for interaction (bypasses ItemAdd)
    InvisibleButton(label, Canvas);
    
    if (IsItemHovered())
        hovered = true;

    RenderFrame(bb.Min, bb.Max, GetColorU32(ImGuiCol_FrameBg, 1), true, Style.FrameRounding);

    for (int i = 0; i <= Canvas.x; i += (Canvas.x / 4)) {
        DrawList->AddLine(ImVec2(bb.Min.x + i, bb.Min.y),
            ImVec2(bb.Min.x + i, bb.Max.y),
            GetColorU32(ImGuiCol_TextDisabled));
    }
    for (int i = 0; i <= Canvas.y; i += (Canvas.y / 4)) {
        DrawList->AddLine(ImVec2(bb.Min.x, bb.Min.y + i),
            ImVec2(bb.Max.x, bb.Min.y + i),
            GetColorU32(ImGuiCol_TextDisabled));
    }

    ImVec2 Q[4] = { {0, 0}, {P[0], P[1]}, {P[2], P[3]}, {1, 1} };
    ImVec2 results[SMOOTHNESS + 1];
    bezier_table<SMOOTHNESS>(Q, results);

    {
        ImVec2 mouse = GetIO().MousePos, pos[2];
        float distance[2];

        for (int i = 0; i < 2; ++i) {
            pos[i] = ImVec2(P[i * 2 + 0], 1 - P[i * 2 + 1]) * (bb.Max - bb.Min) + bb.Min;
            distance[i] = (pos[i].x - mouse.x) * (pos[i].x - mouse.x) +
                (pos[i].y - mouse.y) * (pos[i].y - mouse.y);
        }

        int selected = distance[0] < distance[1] ? 0 : 1;
        if (distance[selected] < (6 * GRAB_RADIUS * 6 * GRAB_RADIUS)) {
            SetTooltip("(%4.3f, %4.3f)", P[selected * 2 + 0], P[selected * 2 + 1]);

            if (IsMouseClicked(0) || IsMouseDragging(0)) {
                float canvasScale = 1.f;
                float& px = (P[selected * 2 + 0] +=
                    GetIO().MouseDelta.x / (Canvas.x * canvasScale));
                float& py = (P[selected * 2 + 1] -=
                    GetIO().MouseDelta.y / (Canvas.y * canvasScale));

                if (AREA_CONSTRAINED) {
                    px = (px < 0 ? 0 : (px > 1 ? 1 : px));
                    py = (py < 0 ? 0 : (py > 1 ? 1 : py));
                }
                changed = true;
            }
        }
    }

    {
        ImColor color(GetStyle().Colors[ImGuiCol_PlotLines]);
        for (int i = 0; i < (int)SMOOTHNESS; ++i) {
            ImVec2 p = { results[i + 0].x, 1 - results[i + 0].y };
            ImVec2 q = { results[i + 1].x, 1 - results[i + 1].y };
            ImVec2 r(p.x * (bb.Max.x - bb.Min.x) + bb.Min.x,
                     p.y * (bb.Max.y - bb.Min.y) + bb.Min.y);
            ImVec2 s(q.x * (bb.Max.x - bb.Min.x) + bb.Min.x,
                     q.y * (bb.Max.y - bb.Min.y) + bb.Min.y);
            DrawList->AddLine(r, s, color, CURVE_WIDTH);
        }
    }

    float luma = IsItemActive() || IsItemHovered() ? 0.5f : 1.0f;
    ImVec4 pink(1.00f, 0.00f, 0.75f, luma), cyan(0.00f, 0.75f, 1.00f, luma);
    ImVec2 p1 = ImVec2(P[0], 1 - P[1]) * (bb.Max - bb.Min) + bb.Min;
    ImVec2 p2 = ImVec2(P[2], 1 - P[3]) * (bb.Max - bb.Min) + bb.Min;
    ImVec4 white(GetStyle().Colors[ImGuiCol_Text]);
    DrawList->AddLine(ImVec2(bb.Min.x, bb.Max.y), p1, ImColor(white), LINE_WIDTH);
    DrawList->AddLine(ImVec2(bb.Max.x, bb.Min.y), p2, ImColor(white), LINE_WIDTH);
    DrawList->AddCircleFilled(p1, GRAB_RADIUS, ImColor(white));
    DrawList->AddCircleFilled(p1, GRAB_RADIUS - GRAB_BORDER, ImColor(pink));
    DrawList->AddCircleFilled(p2, GRAB_RADIUS, ImColor(white));
    DrawList->AddCircleFilled(p2, GRAB_RADIUS - GRAB_BORDER, ImColor(cyan));

    return changed;
}

} // namespace ImGui
