#include "newtype/render/MaterialCallableTable.h"
#include "cinder/Log.h"

namespace newtype::render {

//==============================================================================
// MaterialCallableTable — Singleton implementation
//==============================================================================

MaterialCallableTable& MaterialCallableTable::instance() {
    static MaterialCallableTable table;
    return table;
}

uint MaterialCallableTable::registerCallable(
    const std::string& name, std::any resolve) {

    if (_nextCustomTypeId >= kMaxTypes) {
        CI_LOG_E("MaterialCallableTable full (" << kMaxTypes
                  << "), cannot register '" << name << "'");
        return ~0u;
    }

    if (hasType(_nextCustomTypeId)) {
        CI_LOG_E("MaterialCallableTable type ID " << _nextCustomTypeId
                  << " already registered");
        return ~0u;
    }

    uint type_id = _nextCustomTypeId++;
    MaterialCallableEntry entry;
    entry.name    = name;
    entry.type_id = type_id;
    entry.resolve = std::move(resolve);

    _typeToIndex[type_id] = _entries.size();
    _entries.push_back(std::move(entry));

    CI_LOG_V("Registered custom callable '" << name
             << "' with type ID " << type_id);
    return type_id;
}

void MaterialCallableTable::registerBuiltin(
    uint type_id, const std::string& name, std::any resolve) {

    if (type_id >= kFirstCustomTypeId) {
        CI_LOG_E("registerBuiltin: type_id " << type_id
                  << " is in custom range, use registerCallable instead");
        return;
    }

    if (hasType(type_id)) {
        CI_LOG_W("MaterialCallableTable: built-in type " << type_id
                 << " already registered, replacing");
        size_t idx = _typeToIndex[type_id];
        _entries[idx].resolve = std::move(resolve);
        _entries[idx].name = name;
        return;
    }

    MaterialCallableEntry entry;
    entry.name    = name;
    entry.type_id = type_id;
    entry.resolve = std::move(resolve);

    _typeToIndex[type_id] = _entries.size();
    _entries.push_back(std::move(entry));

    CI_LOG_V("Registered built-in callable '" << name
             << "' with type ID " << type_id);
}

bool MaterialCallableTable::hasType(uint type_id) const {
    return _typeToIndex.find(type_id) != _typeToIndex.end();
}

bool MaterialCallableTable::isCustomType(uint type_id) const {
    return type_id >= kFirstCustomTypeId;
}

void MaterialCallableTable::clearCustom() {
    // Remove entries with custom type IDs (13+)
    _entries.erase(
        std::remove_if(_entries.begin(), _entries.end(),
            [](const MaterialCallableEntry& e) {
                return e.type_id >= kFirstCustomTypeId;
            }),
        _entries.end());

    // Rebuild index
    _typeToIndex.clear();
    for (size_t i = 0; i < _entries.size(); ++i) {
        _typeToIndex[_entries[i].type_id] = i;
    }

    _nextCustomTypeId = kFirstCustomTypeId;
    CI_LOG_V("MaterialCallableTable: cleared custom callables");
}

void MaterialCallableTable::clearAll() {
    _entries.clear();
    _typeToIndex.clear();
    _nextCustomTypeId = kFirstCustomTypeId;
}

} // namespace newtype::render
