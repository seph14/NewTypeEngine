#pragma once
#include "cinder/Json.h"
#include "cinder/Log.h"
#include <filesystem>

namespace newtype {
	namespace util {

		// Write json to path with a rolling single backup: when the existing
		// file's parsed content differs from the new json (or fails to parse),
		// it is first copied to <stem>_bak.json next to it before being
		// overwritten. Unchanged content skips the write entirely — repeated
		// saves don't touch the file's timestamp or churn the backup.
		inline void writeJsonWithBackup(const std::filesystem::path& path, const ci::Json& json) {
			std::error_code ec;
			if (std::filesystem::exists(path, ec)) {
				bool unchanged = false;
				try {
					unchanged = ci::loadJson(path) == json;
				} catch (...) {
				}
				if (unchanged)
					return;
				auto bak = path.parent_path() /
					(path.stem().string() + "_bak" + path.extension().string());
				std::filesystem::copy_file(path, bak,
					std::filesystem::copy_options::overwrite_existing, ec);
				if (ec)
					CI_LOG_W("Failed to back up " << path << " to " << bak
						<< ": " << ec.message());
			}
			ci::writeJson(path, json);
		}

	}
}
