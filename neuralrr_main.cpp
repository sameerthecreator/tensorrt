#include <windows.h>
#include <d3d11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <NvInfer.h>
#include <reshade.hpp>
#include <fstream>
#include <vector>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {}
} g_logger;

IRuntime* g_runtime = nullptr;
ICudaEngine* g_engine = nullptr;
IExecutionContext* g_context = nullptr;
cudaStream_t g_stream = nullptr;

void* d_input = nullptr;
void* d_output = nullptr;

const int IN_W = 768;
const int IN_H = 432;
const int IN_C = 7;

// Load pre-compiled sharp.engine
void init_tensorrt() {
    std::ifstream file("sharp.engine", std::ios::binary);
    if (!file.good()) return;

    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);

    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);

    g_runtime = createInferRuntime(g_logger);
    g_engine = g_runtime->deserializeCudaEngine(engine_data.data(), size);
    g_context = g_engine->createExecutionContext();

    cudaStreamCreate(&g_stream);
    cudaMalloc(&d_input, 1 * IN_C * IN_H * IN_W * sizeof(float));
    cudaMalloc(&d_output, 1 * 3 * IN_H * IN_W * sizeof(float));

    g_context->setInputShape("g_buffer", Dims4{1, IN_C, IN_H, IN_W});
    g_context->setTensorAddress("g_buffer", d_input);
    g_context->setTensorAddress("rendered_frame", d_output);
}

// Hook ReShade Present Call
static void on_present(reshade::api::command_queue* queue, reshade::api::swapchain* swapchain) {
    if (!g_context) return;

    // 1. Enqueue TensorRT async inference on GPU stream
    g_context->enqueueV3(g_stream);
    cudaStreamSynchronize(g_stream);

    // 2. Output is ready in d_output for the swapchain render target
}

extern "C" __declspec(dllexport) const char* NAME = "NeuralRR TensorRT Addon";
extern "C" __declspec(dllexport) const char* DESCRIPTION = "Real-time AI post-processing shader";

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID) {
    switch (reason) {
        case DLL_PROCESS_ATTACH:
            if (!reshade::register_addon(hinst)) return FALSE;
            init_tensorrt();
            reshade::register_event<reshade::addon_event::present>(on_present);
            break;
        case DLL_PROCESS_DETACH:
            reshade::unregister_addon(hinst);
            if (d_input) cudaFree(d_input);
            if (d_output) cudaFree(d_output);
            if (g_stream) cudaStreamDestroy(g_stream);
            delete g_context;
            delete g_engine;
            delete g_runtime;
            break;
    }
    return TRUE;
}

