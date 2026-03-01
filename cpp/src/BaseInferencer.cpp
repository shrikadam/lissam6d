#include "BaseInferencer.hpp"
#include <stdexcept>

BaseInferencer::BaseInferencer(const std::string& model_path, bool use_fp16) 
    : env(ORT_LOGGING_LEVEL_WARNING, "SAM6D_Tracker") {
    
    // 1. Threading and Graph Optimizations
    session_options.SetIntraOpNumThreads(1); // Usually 1 is best when offloading to GPU
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. Append CUDA Execution Provider
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    
    // EXHAUSTIVE tells cuDNN to aggressively benchmark kernels on startup 
    // to find the absolute fastest math operations for your specific Jetson.
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
    
    // If you ever want to use standard CPU fallback, comment this line out.
    session_options.AppendExecutionProvider_CUDA(cuda_options);

    // 3. Load the Model into VRAM
    try {
        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
        std::cout << "Successfully loaded model: " << model_path << " onto CUDA." << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime Error: " << e.what() << std::endl;
        throw std::runtime_error("Failed to initialize ONNX Session.");
    }

    // 4. Dynamically extract Input/Output names so we don't have to hardcode them
    size_t num_inputs = session->GetInputCount();
    for (size_t i = 0; i < num_inputs; i++) {
        // In newer ORT versions, GetInputNameAllocated is required over GetInputName
        Ort::AllocatedStringPtr name_ptr = session->GetInputNameAllocated(i, allocator);
        input_names.push_back(strdup(name_ptr.get())); 
        
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_shapes.push_back(tensor_info.GetShape());
    }

    size_t num_outputs = session->GetOutputCount();
    for (size_t i = 0; i < num_outputs; i++) {
        Ort::AllocatedStringPtr name_ptr = session->GetOutputNameAllocated(i, allocator);
        output_names.push_back(strdup(name_ptr.get()));
    }
}

void BaseInferencer::printModelInfo() {
    std::cout << "\n--- Model I/O Info ---" << std::endl;
    for (size_t i = 0; i < input_names.size(); i++) {
        std::cout << "Input [" << i << "]: " << input_names[i] << " | Shape: [";
        for (auto dim : input_shapes[i]) std::cout << dim << " ";
        std::cout << "]" << std::endl;
    }
}