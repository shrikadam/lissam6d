#include "BaseInferencer.hpp"
#include <iostream>

int main() {
    std::cout << "Starting SAM6D Edge Tracker C++ Engine..." << std::endl;
    
    try {
        // Point this to your exported DINOv2 FP16 model
        // Make sure the path is correct relative to where you run the executable!
        BaseInferencer engine("../models/dinov2_vits14_fp16.onnx", true);
        engine.printModelInfo();
        
    } catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Engine initialized successfully!" << std::endl;
    return 0;
}