#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <iostream>

class BaseInferencer {
protected:
    // ONNX Runtime Core Components
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    // Model Metadata
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;

public:
    BaseInferencer(const std::string& model_path, bool use_fp16 = true);
    virtual ~BaseInferencer() = default;

    // Utility to print model I/O info for debugging
    void printModelInfo(); 
};