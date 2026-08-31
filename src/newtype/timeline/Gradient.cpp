#include "newtype/timeline/Gradient.h"
#include "imgui/imgui_internal.h"
#include "imgui/imgui.h"

using namespace ci;
using namespace std;

namespace newtype::timeline {

static const float GRADIENT_BAR_WIDGET_HEIGHT = 25;
static const float GRADIENT_BAR_EDITOR_HEIGHT = 40;
static const float GRADIENT_MARK_DELETE_DIFFY = 40;

Gradient::Gradient() {
    addMark(0.0f, ImColor(1.0f, 1.0f, 1.0f));
    addMark(1.0f, ImColor(1.0f, 1.0f, 1.0f));
}

Gradient::Gradient(const Gradient& other) {
    for (auto* mark : other.m_marks) {
        auto* m = new GradientMark();
        m->position = mark->position;
        m->color = mark->color;
        m_marks.push_back(m);
    }
    refreshCache();
}

Gradient::~Gradient() {
    for (GradientMark* mark : m_marks)
        delete mark;
}

void Gradient::getColorAt(float position, ci::ColorA* color) const {
    position = ImClamp(position, 0.0f, 1.0f);
    int cachePos = (int)(position * 255);
    *color = m_cachedValues[cachePos];
}

void Gradient::addMark(float position, ImColor const color) {
    position = ImClamp(position, 0.0f, 1.0f);
    auto* newMark = new GradientMark();
    newMark->position = position;
    newMark->color[0] = color.Value.x;
    newMark->color[1] = color.Value.y;
    newMark->color[2] = color.Value.z;
    newMark->color[3] = color.Value.w;
    m_marks.push_back(newMark);
    refreshCache();
}

void Gradient::removeMark(GradientMark* mark) {
    m_marks.remove(mark);
    delete mark;
    refreshCache();
}

void Gradient::computeColorAt(float position, ci::ColorA* color) const {
    position = ImClamp(position, 0.0f, 1.0f);

    GradientMark* lower = nullptr;
    GradientMark* upper = nullptr;

    for (GradientMark* mark : m_marks) {
        if (mark->position < position) {
            if (!lower || lower->position < mark->position)
                lower = mark;
        }
        if (mark->position >= position) {
            if (!upper || upper->position > mark->position)
                upper = mark;
        }
    }

    if (upper && !lower) lower = upper;
    else if (!upper && lower) upper = lower;
    else if (!lower && !upper) {
        *color = ColorA::black();
        return;
    }

    if (upper == lower) {
        *color = upper->color;
    } else {
        float distance = upper->position - lower->position;
        float delta = (position - lower->position) / distance;
        color->r = (1.0f - delta) * lower->color.r + delta * upper->color.r;
        color->g = (1.0f - delta) * lower->color.g + delta * upper->color.g;
        color->b = (1.0f - delta) * lower->color.b + delta * upper->color.b;
        color->a = (1.0f - delta) * lower->color.a + delta * upper->color.a;
    }
}

void Gradient::refreshCache() {
    m_marks.sort([](const GradientMark* a, const GradientMark* b) {
        return a->position < b->position;
    });
    for (int i = 0; i < 256; ++i)
        computeColorAt(i / 255.0f, &m_cachedValues[i]);
}

luisa::vector<luisa::half4> Gradient::genData() {
    using namespace luisa;
    luisa::vector<half4> data(256u * 2u);
    for (uint32_t i = 0; i < 256u; i++) {
        auto& col = m_cachedValues[i];
        auto px = make_half4(
            static_cast<half>(col.r),
            static_cast<half>(col.g),
            static_cast<half>(col.b),
            static_cast<half>(col.a)
        );

        data[i] = px;
        data[i+256u] = px;
    }
    return data;
}

ci::Surface8uRef Gradient::genSurface() {
    auto surf = Surface8u::create(256, 2, true);
    auto iter = surf->getIter();
    while (iter.line()) {
        while (iter.pixel()) {
            auto& col = m_cachedValues[iter.x()];
            iter.r() = (uint8_t)(255 * col.r);
            iter.g() = (uint8_t)(255 * col.g);
            iter.b() = (uint8_t)(255 * col.b);
            iter.a() = (uint8_t)(255 * col.a);
        }
    }
    return surf;
}

ci::Json Gradient::toJson() const {
    auto file = ci::Json::array();
    for (auto* mark : m_marks) {
        auto obj = ci::Json::object();
        obj["position"] = mark->position;
        auto colorArr = ci::Json::array();
        colorArr.push_back(mark->color.r);
        colorArr.push_back(mark->color.g);
        colorArr.push_back(mark->color.b);
        colorArr.push_back(mark->color.a);
        obj["color"] = colorArr;
        file.push_back(obj);
    }
    return file;
}

void Gradient::fromJson(const ci::Json& j) {
    for (auto* mark : m_marks) delete mark;
    m_marks.clear();

    for (size_t i = 0; i < j.size(); ++i) {
        auto item = j[i];
        auto* mark = new GradientMark();
        mark->position = item["position"];
        mark->color = ColorA(
            (float)item["color"][0], (float)item["color"][1],
            (float)item["color"][2], (float)item["color"][3]);
        m_marks.push_back(mark);
    }
    refreshCache();
}

} // namespace newtype::timeline

// ============================================================================
// ImGui Gradient widgets
// ============================================================================

using namespace newtype::timeline;

namespace ImGui {

static void DrawGradientBar(Gradient* gradient, ImVec2 const& bar_pos,
                            float maxWidth, float height) {
    float barBottom = bar_pos.y + height;
    GradientMark* prevMark = nullptr;
    ImDrawList* draw_list = GetWindowDrawList();

    // Draw border/background
    draw_list->AddRectFilled(ImVec2(bar_pos.x - 2, bar_pos.y - 2),
        ImVec2(bar_pos.x + maxWidth + 2, barBottom + 2),
        IM_COL32(100, 100, 100, 255));

    if (gradient->getMarks().empty()) {
        draw_list->AddRectFilled(ImVec2(bar_pos.x, bar_pos.y),
            ImVec2(bar_pos.x + maxWidth, barBottom),
            IM_COL32(255, 255, 255, 255));
        //SetCursorScreenPos(ImVec2(bar_pos.x, bar_pos.y + height + 10.0f));
        return;
    }

    // Draw gradient segments
    for (auto markIt = gradient->getMarks().begin();
         markIt != gradient->getMarks().end(); ++markIt) {
        GradientMark* mark = *markIt;
        float toX = bar_pos.x + mark->position * maxWidth;

        // Determine segment start position
        float fromX = prevMark ? bar_pos.x + prevMark->position * maxWidth : bar_pos.x;

        ImVec4 colorA, colorB;
        if (prevMark) {
            colorA = ImVec4(prevMark->color[0], prevMark->color[1], prevMark->color[2], prevMark->color[3]);
        } else {
            colorA = ImVec4(mark->color[0], mark->color[1], mark->color[2], mark->color[3]);
        }
        colorB = ImVec4(mark->color[0], mark->color[1], mark->color[2], mark->color[3]);

        ImU32 colorAU32 = ColorConvertFloat4ToU32(colorA);
        ImU32 colorBU32 = ColorConvertFloat4ToU32(colorB);

        // Draw gradient segment (or solid color if from/to are same)
        if (fromX < toX) {
            draw_list->AddRectFilledMultiColor(
                ImVec2(fromX, bar_pos.y),
                ImVec2(toX, barBottom),
                colorAU32, colorBU32, colorBU32, colorAU32);
        }

        prevMark = mark;
    }

    // Fill remaining area with last mark's color
    if (prevMark && prevMark->position < 1.f) {
        float lastX = bar_pos.x + prevMark->position * maxWidth;
        ImVec4 lastColor(prevMark->color[0], prevMark->color[1], prevMark->color[2], prevMark->color[3]);
        ImU32 lastColU32 = ColorConvertFloat4ToU32(lastColor);
        draw_list->AddRectFilled(ImVec2(lastX, bar_pos.y),
            ImVec2(bar_pos.x + maxWidth, barBottom),
            lastColU32);
    }

    //SetCursorScreenPos(ImVec2(bar_pos.x, bar_pos.y + height + 10.0f));
}

static void DrawGradientMarks(Gradient* gradient,
                              GradientMark*& draggingMark,
                              GradientMark*& selectedMark,
                              ImVec2 const& bar_pos,
                              float maxWidth, float height) {
    float barBottom = bar_pos.y + height;
    ImDrawList* draw_list = GetWindowDrawList();
    int markIndex = 0;

    for (auto markIt = gradient->getMarks().begin();
         markIt != gradient->getMarks().end(); ++markIt, ++markIndex) {
        GradientMark* mark = *markIt;

        if (!selectedMark) selectedMark = mark;

        float to = bar_pos.x + mark->position * maxWidth;

        draw_list->AddTriangleFilled(
            ImVec2(to, bar_pos.y + (height - 6)),
            ImVec2(to - 6, barBottom),
            ImVec2(to + 6, barBottom),
            IM_COL32(100, 100, 100, 255));

        draw_list->AddRectFilled(ImVec2(to - 6, barBottom),
            ImVec2(to + 6, bar_pos.y + (height + 12)),
            IM_COL32(100, 100, 100, 255), 1.0f, ImDrawFlags_None);

        draw_list->AddRectFilled(ImVec2(to - 5, bar_pos.y + (height + 1)),
            ImVec2(to + 5, bar_pos.y + (height + 11)),
            IM_COL32(0, 0, 0, 255), 1.0f, ImDrawFlags_None);

        if (selectedMark == mark) {
            draw_list->AddTriangleFilled(
                ImVec2(to, bar_pos.y + (height - 3)),
                ImVec2(to - 4, barBottom + 1),
                ImVec2(to + 4, barBottom + 1),
                IM_COL32(0, 255, 0, 255));
            draw_list->AddRect(
                ImVec2(to - 5, bar_pos.y + (height + 1)),
                ImVec2(to + 5, bar_pos.y + (height + 11)),
                IM_COL32(0, 255, 0, 255), 1.0f, ImDrawFlags_None);
        }

        ImVec4 colorB = {mark->color[0], mark->color[1], mark->color[2], mark->color[3]};
        ImU32 colorBU32 = ColorConvertFloat4ToU32(colorB);
        draw_list->AddRectFilledMultiColor(
            ImVec2(to - 3, bar_pos.y + (height + 3)),
            ImVec2(to + 3, bar_pos.y + (height + 9)),
            colorBU32, colorBU32, colorBU32, colorBU32);

        // Use unique ID based on mark pointer
        PushID(mark);
        SetCursorScreenPos(ImVec2(to - 6, barBottom));
        InvisibleButton("mark", ImVec2(12, 12));
        PopID();

        if (IsItemHovered() && IsMouseClicked(0)) {
            selectedMark = mark;
            draggingMark = mark;
        }
    }

    SetCursorScreenPos(ImVec2(bar_pos.x, bar_pos.y + height + 20.0f));
}

bool GradientButton(Gradient* gradient) {
    if (!gradient) return false;

    ImVec2 widget_pos = GetCursorScreenPos();
    float maxWidth = ImMax(250.0f, GetContentRegionAvail().x - 100.0f);
    bool clicked = InvisibleButton("gradient_bar",
        ImVec2(maxWidth, GRADIENT_BAR_WIDGET_HEIGHT));

    DrawGradientBar(gradient, widget_pos, maxWidth, GRADIENT_BAR_WIDGET_HEIGHT);
    return clicked;
}

bool GradientEditor(Gradient* gradient,
                    GradientMark*& draggingMark,
                    GradientMark*& selectedMark) {
    if (!gradient) return false;

    bool modified = false;
    ImVec2 bar_pos = GetCursorScreenPos();
    bar_pos.x += 10;
    float maxWidth = GetContentRegionAvail().x - 20;

    InvisibleButton("gradient_editor_bar",
        ImVec2(maxWidth, GRADIENT_BAR_EDITOR_HEIGHT));

    if (IsItemHovered() && IsMouseClicked(0)) {
        float pos = (GetIO().MousePos.x - bar_pos.x) / maxWidth;
        ColorA newMarkCol;
        gradient->getColorAt(pos, &newMarkCol);
        gradient->addMark(pos, ImColor(newMarkCol[0], newMarkCol[1], newMarkCol[2]));
    }

    DrawGradientBar(gradient, bar_pos, maxWidth, GRADIENT_BAR_EDITOR_HEIGHT);
    DrawGradientMarks(gradient, draggingMark, selectedMark,
                      bar_pos, maxWidth, GRADIENT_BAR_EDITOR_HEIGHT);

    if (!IsMouseDown(0) && draggingMark)
        draggingMark = nullptr;

    if (IsMouseDragging(0) && draggingMark) {
        float increment = GetIO().MouseDelta.x / maxWidth;
        bool insideZone = (GetIO().MousePos.x > bar_pos.x) &&
                          (GetIO().MousePos.x < bar_pos.x + maxWidth);

        if (increment != 0.0f && insideZone) {
            draggingMark->position += increment;
            draggingMark->position = ImClamp(draggingMark->position, 0.0f, 1.0f);
            gradient->refreshCache();
            modified = true;
        }

        float diffY = GetIO().MousePos.y - (bar_pos.y + GRADIENT_BAR_EDITOR_HEIGHT);
        if (diffY >= GRADIENT_MARK_DELETE_DIFFY) {
            gradient->removeMark(draggingMark);
            draggingMark = nullptr;
            selectedMark = nullptr;
            modified = true;
        }
    }

    if (!selectedMark && !gradient->getMarks().empty())
        selectedMark = gradient->getMarks().front();

    if (selectedMark) {
        bool colorModified = ColorPicker4("", &selectedMark->color.r);
        if (colorModified) {
            modified = true;
            gradient->refreshCache();
        }
    }

    return modified;
}

} // namespace ImGui
