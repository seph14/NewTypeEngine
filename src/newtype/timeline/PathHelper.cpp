#include "newtype/timeline/PathHelper.h"
#include "cinder/app/App.h"
#include "cinder/gl/gl.h"
#if NT_ENABLE_TIMELINE_EDITOR
#include "cinder/CinderImGui.h"
#endif

using namespace ci;
using namespace std;

#if NT_ENABLE_TIMELINE_EDITOR
static const int SAMPLE_POINTS = 64;
#endif

namespace newtype::timeline {

#if NT_ENABLE_TIMELINE_EDITOR
template<int D>
void PathHelper<D>::bakePoints() {
    mBakedPos.clear();
    for (int i = 0; i < SAMPLE_POINTS; i++) {
        float rr = glm::clamp((float)i / (SAMPLE_POINTS - 1), 0.f, 1.f);
        mBakedPos.push_back(mCurve.getPosition(rr));
    }
    if constexpr (D == 4) {
        for (auto& v : mBakedPos) v[3] = 1.f;
    }
    mParticleVbo = gl::Vbo::create(GL_ARRAY_BUFFER, mBakedPos, GL_STREAM_DRAW);
    geom::BufferLayout particleLayout;
    particleLayout.append(geom::Attrib::POSITION, D, sizeof(vecP), 0);
    auto mesh = gl::VboMesh::create(SAMPLE_POINTS, GL_POINTS,
        {{particleLayout, mParticleVbo}});
    mParticleBatch = gl::Batch::create(mesh,
        gl::getStockShader(gl::ShaderDef().color()));
}
#endif

template<int D>
PathHelper<D>::PathHelper(std::vector<vecP> path) {
    setPathData(path);
}

template<int D>
PathHelper<D>::PathHelper() = default;

template<int D>
void PathHelper<D>::getCurveData(float ratio, vecP* pos, vecP* dir) {
    *pos = mCurve.getPosition(glm::clamp(ratio, 0.f, 1.f));
    bool reverse = (ratio >= 1.f);
    auto alter = mCurve.getPosition(ratio + (reverse ? -1.f : 1.f) / 768.f);
    if constexpr (D > 3)
        *dir = (reverse ? -1.f : 1.f) * vec4(glm::normalize(vec3(alter - *pos)), 0.f);
    else
        *dir = (reverse ? -1.f : 1.f) * glm::normalize(alter - *pos);
}

template<int D>
ci::mat4 PathHelper<D>::getRotation(float ratio) const {
    ratio = mCurve.getTime(glm::clamp(ratio, 0.f, 1.f) * mLength);
    //ratio = mEasing.sample(ratio).y;
    vecP pos = mCurve.getPosition(ratio);
    bool reverse = (ratio >= 767.f / 768.f);
    vecP alter = mCurve.getPosition(ratio + (reverse ? -1.f : 1.f) / 768.f);
    vec3 dir = (reverse ? -1.f : 1.f) * glm::normalize(vec3(alter - pos));
    quat rot = glm::quatLookAt(dir, vec3(0.f, 1.f, 0.f));
    return glm::mat4_cast(rot);
}

template<int D>
typename PathHelper<D>::vecP PathHelper<D>::getPosition(float ratio) const {
    ratio = mCurve.getTime(glm::clamp(ratio, 0.f, 1.f) * mLength);
    return mCurve.getPosition(mEasing.sample(ratio).y);
}

template<int D>
ci::mat4 PathHelper<D>::getTransform(float ratio, bool includeRotation) const {
    ratio = mCurve.getTime(glm::clamp(ratio, 0.f, 1.f) * mLength);
    //ratio = mEasing.sample(ratio).y;
    vecP pos = mCurve.getPosition(ratio);
    float scale = 1.f;
    if constexpr (D > 3) scale = pos[D - 1];

    if (!includeRotation)
        return glm::translate(vec3(pos)) * glm::scale(vec3(scale));

    bool reverse = (ratio >= 767.f / 768.f);
    auto alter = mCurve.getPosition(ratio + (reverse ? -1.f : 1.f) / 768.f);
    vec3 dir = (reverse ? -1.f : 1.f) * glm::normalize(vec3(alter - vecP(pos)));
    quat rot = glm::quatLookAt(dir, vec3(0.f, 1.f, 0.f));
    return glm::translate(vec3(pos)) * glm::mat4_cast(rot) * glm::scale(vec3(scale));
}

template<int D>
void PathHelper<D>::getNormalizedData(float ratio, vecP* pos, vecP* dir) {
    ratio = mCurve.getTime(glm::clamp(ratio, 0.f, 1.f) * mLength);
    //ratio = mEasing.sample(ratio).y;
    *pos = mCurve.getPosition(ratio);
    bool reverse = (ratio >= 1.f);
    auto alter = mCurve.getPosition(ratio + (reverse ? -1.f : 1.f) / 768.f);
    if constexpr (D > 3)
        *dir = (reverse ? -1.f : 1.f) * vec4(glm::normalize(vec3(alter - *pos)), 1.f);
    else
        *dir = (reverse ? -1.f : 1.f) * glm::normalize(alter - *pos);
}

template<int D>
ci::vec3 PathHelper<D>::getNormalizedDirection(float ratio) const {
    ratio = mCurve.getTime(glm::clamp(ratio, 0.f, 1.f) * mLength);
    //ratio = mEasing.sample(ratio).y;
    vecP pos = mCurve.getPosition(ratio);
    bool reverse = (ratio >= 1.f);
    vecP alter = mCurve.getPosition(ratio + (reverse ? -1.f : 1.f) / 768.f);
    float direc = reverse ? -1.f : 1.f;
    return direc * glm::normalize(vec3(alter - pos));
}

template<int D>
void PathHelper<D>::setPathData(std::vector<vecP> path) {
    if (path.size() > 3) {
        mCurve = BSpline<D, float>(path, 3, false, true);
        mLength = mCurve.getLength(0.f, 1.f);
#if NT_ENABLE_TIMELINE_EDITOR
        bakePoints();
#endif
        vec3 dir = getNormalizedDirection(0.f);
        mRootPos = mCurve.getPosition(0.f);
        mRootRot = glm::quatLookAt(dir, vec3(0.f, 1.f, 0.f));
    }
}

#if NT_ENABLE_TIMELINE_EDITOR
template<int D>
void PathHelper<D>::updateSelf() {
    mLength = mCurve.getLength(0.f, 1.f);
    bakePoints();
    vec3 dir = getNormalizedDirection(0.f);
    mRootPos = mCurve.getPosition(0.f);
    mRootRot = glm::quatLookAt(dir, vec3(0.f, 1.f, 0.f));
}

template<int D>
void PathHelper<D>::drawDebugPoints(float ratio, ci::Color controlPnt, ci::Color currPos) {
    gl::ScopedModelMatrix scpModel;
    gl::setModelMatrix(mat4());
    gl::pointSize(16.f);

    {
        gl::ScopedColor scpColor(
            (mNodeIdx >= 0) ? Color(0.f, 0.f, 1.f) : Color::gray(.75f));
        mParticleBatch->draw();
    }

    {
        gl::ScopedGlslProg scpGlsl(gl::getStockShader(gl::ShaderDef().color()));
        if (mNodeIdx >= 0 && ratio >= 0.f && ratio <= 1.f) {
            ratio = mCurve.getTime(glm::clamp(ratio, 0.f, 1.f) * mLength);
            vecP pos = mCurve.getPosition(ratio);
            gl::ScopedLineWidth scpLine(2.f);
            gl::ScopedColor scpColor(currPos);
            float s = 1.f;
            if constexpr (D > 3) s = pos[D - 1];
            gl::drawStrokedCube(vec3(pos), s * vec3(2.6f));
        }
        {
            gl::ScopedColor scpColor(controlPnt);
            for (int i = 0; i < mCurve.getNumControlPoints(); i++) {
                vecP p = mCurve.getControlPoint(i);
                float s = 1.f;
                if constexpr (D > 3) s = p[D - 1];
                gl::drawSphere(vec3(p), s * .125f);
            }
        }
    }
}

template<int D>
bool PathHelper<D>::drawUi(const std::string& name) {
    ImGui::ScopedId scpId(name.c_str());
    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.f, 5.f));
    ImGui::Text("%s-Path", name.c_str());

    bool changed = ImGui::SliderInt("Node",
        &mNodeIdx, -1, mCurve.getNumControlPoints() - 1);

    ImGui::Separator();
    ImGui::Dummy(ImVec2(0.f, 5.f));

    if (mNodeIdx >= 0) {
        vecP p = mCurve.getControlPoint(mNodeIdx);
        bool updated = ImGui::InputFloat3("Pos", &p.x);
        if constexpr (D > 3) {
            float s = p[3];
            updated |= ImGui::InputFloat("Scale", &s);
            p[3] = s;
        }
        if (updated) {
            mCurve.setControlPoint(mNodeIdx, p);
            updateSelf();
        }

        if (mCurve.getNumControlPoints() > 4) {
            if (ImGui::Button("Remove Current Control Point")) {
                vector<vecP> data;
                for (int i = 0; i < mCurve.getNumControlPoints(); i++) {
                    if (i == mNodeIdx) continue;
                    data.push_back(mCurve.getControlPoint(i));
                }
                setPathData(data);
            }
        }

        ImGui::Separator();
        ImGui::Dummy(ImVec2(0.f, 5.f));

        if (ImGui::Button("Insert Ctrl Pnt After")) {
            vector<vecP> data;
            for (int i = 0; i < mCurve.getNumControlPoints(); i++) {
                data.push_back(mCurve.getControlPoint(i));
                if (i == mNodeIdx && i < mCurve.getNumControlPoints() - 1) {
                    data.push_back(glm::mix(mCurve.getControlPoint(i),
                        mCurve.getControlPoint(i + 1), .5f));
                }
            }
            if (mNodeIdx == mCurve.getNumControlPoints() - 1) {
                vecP c = mCurve.getControlPoint(mNodeIdx);
                vecP p = mCurve.getControlPoint(mNodeIdx - 1);
                data.push_back(2.f * c - p);
            }
            setPathData(data);
        }

        if (ImGui::Button("Insert Ctrl Pnt Before")) {
            vector<vecP> data;
            if (mNodeIdx == 0) {
                vecP c = mCurve.getControlPoint(0);
                vecP p = mCurve.getControlPoint(1);
                data.push_back(2.f * c - p);
            }
            for (int i = 0; i < mCurve.getNumControlPoints(); i++) {
                data.push_back(mCurve.getControlPoint(i));
                if (i == mNodeIdx && i > 0) {
                    data.push_back(glm::mix(mCurve.getControlPoint(i),
                        mCurve.getControlPoint(i - 1), .5f));
                }
            }
            setPathData(data);
        }

        ImGui::Separator();
        ImGui::Dummy(ImVec2(0.f, 5.f));
    }

    if (ImGui::Button("Add Ctrl Pnt")) {
        vector<vecP> data;
        for (int i = 0; i < mCurve.getNumControlPoints(); i++)
            data.push_back(mCurve.getControlPoint(i));
        vecP c = mCurve.getControlPoint(mCurve.getNumControlPoints() - 1);
        vecP p = mCurve.getControlPoint(mCurve.getNumControlPoints() - 2);
        data.push_back(2.f * c - p);
        setPathData(data);
    }

    ImGui::Separator();
    changed |= mEasing.drawUi("Easing");
    ImGui::Dummy(ImVec2(0.f, 5.f));
    return changed;
}
#endif // NT_ENABLE_TIMELINE_EDITOR

template<int D>
ci::Json PathHelper<D>::toJson() const {
    auto file = ci::Json::object();
    file["dim"] = D;
    file["degree"] = mCurve.getDegree();
    auto data = ci::Json::array();
    for (int i = 0; i < mCurve.getNumControlPoints(); i++) {
        vecP p = mCurve.getControlPoint(i);
        auto item = ci::Json::array();
        for (int v = 0; v < D; v++)
            item.push_back(p[v]);
        data.push_back(item);
    }
    file["data"] = data;
    file["easing"] = mEasing.toJson();
    return file;
}

template<int D>
void PathHelper<D>::fromJson(const ci::Json& j) {
    if (j.contains("easing"))
        mEasing.fromJson(j["easing"]);
}

// Explicit instantiation
template class PathHelper<3>;

} // namespace newtype::timeline
