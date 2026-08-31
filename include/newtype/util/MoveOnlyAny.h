#pragma once
#include <memory>
#include <any>

namespace newtype::core {

    // Type-erased wrapper for move-only types (like Shader2D)
    class MoveOnlyAny {
    public:
        MoveOnlyAny() = default;
        MoveOnlyAny(std::nullptr_t) noexcept {}

        template<typename T>
        MoveOnlyAny(T&& value)
            requires (!std::same_as<std::decay_t<T>, MoveOnlyAny>)
        {
            using Td = std::decay_t<T>;
            _ptr = std::make_unique<Model<Td>>(std::forward<T>(value));
        }

        MoveOnlyAny(MoveOnlyAny&& other) noexcept = default;
        MoveOnlyAny& operator=(MoveOnlyAny&& other) noexcept = default;

        // Non-copyable
        MoveOnlyAny(const MoveOnlyAny&) = delete;
        MoveOnlyAny& operator=(const MoveOnlyAny&) = delete;

        [[nodiscard]] explicit operator bool() const noexcept { return _ptr != nullptr; }

        template<typename T>
        [[nodiscard]] T& cast()& {
            auto* model = dynamic_cast<Model<T>*>(_ptr.get());
            if (!model) {
                throw std::bad_any_cast();
            }
            return model->value;
        }

        template<typename T>
        [[nodiscard]] const T& cast() const& {
            auto* model = dynamic_cast<Model<T>*>(_ptr.get());
            if (!model) {
                throw std::bad_any_cast();
            }
            return model->value;
        }

    private:
        struct Concept {
            virtual ~Concept() = default;
        };

        template<typename T>
        struct Model : Concept {
            explicit Model(T&& v) : value(std::forward<T>(v)) {}
            T value;
        };

        std::unique_ptr<Concept> _ptr;
    };
}