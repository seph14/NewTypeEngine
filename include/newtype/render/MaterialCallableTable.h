#pragma once

#include <string>
#include <vector>
#include <any>
#include <cstdint>
#include <unordered_map>

#include "newtype/render/Material.h"

namespace newtype::render {

/// Unsigned int shorthand (consistent with LuisaCompute DSL types)
using uint = std::uint32_t;

//==============================================================================
// Material Callable Table — Per-type dispatch for surface resolution
//==============================================================================

/**
 * @brief Entry in the material callable table.
 *
 * The resolve field stores a type-erased callable (std::any).
 * The actual type is:
 *   std::function<SurfaceData(SurfaceData, Var<MaterialData>,
 *                             Float2, Float2, Float3, Float,
 *                             const BindlessVar&, UInt, UInt)>
 *
 * This type can ONLY be spelled inside device.compile contexts
 * (where Var<MaterialData> is valid). Outside that context,
 * the callable is stored as std::any and cast at dispatch time.
 *
 * Built-in types (below kFirstCustomTypeId): resolve is empty std::any
 * (identity transform). Custom types (kFirstCustomTypeId..kMaxTypes-1):
 * resolve holds the user-defined callable.
 */
struct MaterialCallableEntry {
    std::string        name;       ///< Debug name
    uint               type_id;    ///< MaterialType ID (built-in, or custom >= kFirstCustomTypeId)
    std::any           resolve;    ///< Type-erased callable body (empty = identity)
};

/**
 * @brief Unified registry for material type callables.
 *
 * Stores callable definitions for ALL material types (built-in + custom).
 * At kernel compile time, entries() is iterated with a C++ for-loop
 * to generate $switch/$case dispatch in resolve_surface().
 *
 * Built-in types are registered at startup with identity transforms
 * (empty std::any — dispatch skips them).
 * Custom types (kFirstCustomTypeId..kMaxTypes-1) are registered by the user
 * before buildScene().
 */
class MaterialCallableTable {
public:
    static constexpr uint kMaxTypes = 32u;
    /// Single source of truth: kFirstCustomMaterialType in Material.h, derived
    /// from the last built-in MaterialType. Adding a built-in type shifts the
    /// custom range automatically.
    static constexpr uint kFirstCustomTypeId = kFirstCustomMaterialType;

    /// Register a custom callable (type-erased). Returns assigned type ID, or ~0u on failure.
    /// The std::any MUST contain a SurfaceResolveFn (std::function<SurfaceData(...)>).
    uint registerCallable(const std::string& name, std::any resolve);

    /// Register a built-in callable (called at startup).
    void registerBuiltin(uint type_id, const std::string& name, std::any resolve);

    /// Get all entries (for kernel compile-time $switch generation).
    const std::vector<MaterialCallableEntry>& entries() const { return _entries; }

    /// Query
    [[nodiscard]] bool hasType(uint type_id) const;
    [[nodiscard]] bool isCustomType(uint type_id) const;

    /// Clear only custom types (for DLL reload).
    void clearCustom();

    /// Clear all types.
    void clearAll();

    /// Singleton accessor.
    static MaterialCallableTable& instance();

private:
    MaterialCallableTable() = default;

    std::vector<MaterialCallableEntry> _entries;
    std::unordered_map<uint, size_t>   _typeToIndex;
    uint _nextCustomTypeId = kFirstCustomTypeId;
};

/// Register all 13 built-in material callables (identity transforms).
/// Called at Pipeline::buildScene(). Safe to call multiple times.
void registerBuiltinCallables();

} // namespace newtype::render
