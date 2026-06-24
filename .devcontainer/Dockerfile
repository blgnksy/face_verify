FROM ubuntu:26.04

ENV DEBIAN_FRONTEND=noninteractive
ENV ONNXRUNTIME_VERSION=1.18.0

# Build tools, OpenCV, PAM, and utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    unzip \
    pkg-config \
    libopencv-dev \
    libpam-dev \
    v4l-utils \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Developer tooling — conventional commits, pre-commit hooks, coverage
RUN pip3 install --break-system-packages commitizen pre-commit gcovr

# ONNX Runtime — install from official release tarball
RUN set -eux; \
    wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" -O /tmp/ort.tgz; \
    tar -xzf /tmp/ort.tgz -C /tmp; \
    ORT_DIR="/tmp/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}"; \
    cp "${ORT_DIR}"/lib/libonnxruntime*.so* /usr/local/lib/; \
    cp "${ORT_DIR}"/include/*.h /usr/local/include/; \
    if [ -d "${ORT_DIR}/lib/cmake" ]; then \
        cp -r "${ORT_DIR}"/lib/cmake /usr/local/lib/; \
    fi; \
    ldconfig; \
    rm -rf /tmp/ort.tgz "${ORT_DIR}"

# Write a cmake config so find_package(onnxruntime) works even if the tarball
# didn't ship one (older releases omit it).
RUN mkdir -p /usr/local/lib/cmake/onnxruntime \
    && if [ ! -f /usr/local/lib/cmake/onnxruntime/onnxruntimeConfig.cmake ]; then \
    printf '%s\n' \
        'if(NOT TARGET onnxruntime::onnxruntime)' \
        '  add_library(onnxruntime::onnxruntime SHARED IMPORTED)' \
        '  set_target_properties(onnxruntime::onnxruntime PROPERTIES' \
        '    IMPORTED_LOCATION "/usr/local/lib/libonnxruntime.so"' \
        '    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include"' \
        '  )' \
        'endif()' \
        'set(onnxruntime_FOUND TRUE)' \
    > /usr/local/lib/cmake/onnxruntime/onnxruntimeConfig.cmake; \
    fi

WORKDIR /workspace
