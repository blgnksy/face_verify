/**
 * face_verify/config — configuration file loading and saving.
 * Copyright (C) 2026  Bilgin Aksoy
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

#include "face_verify/types.hpp"

namespace fv {

// Load a key=value config file and apply values to cfg.
// Unknown keys are silently ignored; malformed values throw std::runtime_error.
// Does NOT throw if the file doesn't exist — call only after checking
// fs::exists().
void load_config(Config &cfg, const fs::path &path);

// Write the current config to a file (human-readable, can be used as a
// template).
void save_config(const Config &cfg, const fs::path &path);

} // namespace fv
