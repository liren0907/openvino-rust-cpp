// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <string>
#include <sstream>
#include <mutex>

namespace slog {

class LogStreamEndLine {};

static constexpr LogStreamEndLine endl{};

class LogStream {
private:
    std::stringstream _sstream;
    std::mutex _mutex;

    std::ostream& getStream() {
        return _sstream;
    }

public:
    LogStream() = default;

    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;

    ~LogStream() {
        std::lock_guard<std::mutex> lock(_mutex);
        std::cout << _sstream.str();
        std::cout.flush();
    }

    template <typename T>
    LogStream& operator<<(const T& value) {
        _sstream << value;
        return *this;
    }

    LogStream& operator<<(const LogStreamEndLine&) {
        _sstream << std::endl;
        return *this;
    }
};

class Loggers {
public:
    static LogStream& info() {
        static LogStream instance;
        return instance;
    }

    static LogStream& warning() {
        static LogStream instance;
        return instance;
    }

    static LogStream& error() {
        static LogStream instance;
        return instance;
    }

    static LogStream& debug() {
        static LogStream instance;
        return instance;
    }
};

inline LogStream& info {Loggers::info()};
inline LogStream& warn {Loggers::warning()};
inline LogStream& err {Loggers::error()};
inline LogStream& debug {Loggers::debug()};

} // namespace slog
