#include "newtype/util/CommandBuffer.h"
#include "cinder/Log.h"

namespace newtype::util {
	using namespace luisa;
    CommandBuffer::CommandBuffer(Stream* stream) noexcept
        : _stream{ stream } {
    }

    CommandBuffer::~CommandBuffer() noexcept {
        CI_ASSERT_MSG(_list.empty(),
            "Command buffer not empty when destroyed. "
            "Did you forget to commit?");
    }
}