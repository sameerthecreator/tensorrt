#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <reshade.hpp>
#include <NvInfer.h>
#include <fstream>
#include <vector>
#include <cstdint>

// ============================================================================
// 1. DYNAMIC CUDA & TENSORRT FUNCTION POINTERS (ZERO .LIB FILES NEEDED)
// ============================================================================
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;

typedef cudaError_t (*pfn_cudaMalloc)(void** devPtr, size_t size);
typedef cudaError_t (*pfn_cudaFree)(void* devPtr);
typedef cudaError_t (*pfn_cudaMemcpyAsync)(void* dst, const void* src, size_t count, int kind, cudaStream_t stream);
typedef cudaError_t (*pfn_cudaStreamCreate)(cudaStream_t* pStream);
typedef cudaError_t (*pfn_cudaStreamDestroy)(cudaStream_t stream);
typedef cudaError_t (*pfn_cudaStreamSynchronize)(cudaStream_t stream);
typedef void* (*pfn_createInferRuntime_INTERNAL)(void* logger, int32_t version);

static HMODULE g_hCuda = nullptr;
static HMODULE g_hTrt  = nullptr;

static pfn_cudaMalloc            fn_cudaMalloc = nullptr;
static pfn_cudaFree              fn_cudaFree = nullptr;
static pfn_cudaMemcpyAsync       fn_cudaMemcpyAsync = nullptr;
static pfn_cudaStreamCreate      fn_cudaStreamCreate = nullptr;
static pfn_cudaStreamDestroy     fn_cudaStreamDestroy = nullptr;
static pfn_cudaStreamSynchronize fn_cudaStreamSynchronize = nullptr;
static pfn_createInferRuntime_INTERNAL fn_createInferRuntime = nullptr;

// ============================================================================
// 2. TENSORRT STATE & PERSISTENT GPU BUFFERS
// ============================================================================
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            OutputDebugStringA("[NeuralRR] ");
            OutputDebugStringA(msg);
            OutputDebugStringA("\n");
        }
    }
} g_logger;

static nvinfer1::IRuntime*          g_trtRuntime = nullptr;
static nvinfer1::ICudaEngine*       g_trtEngine  = nullptr;
static nvinfer1::IExecutionContext* g_trtContext = nullptr;
static cudaStream_t                 g_cudaStream = nullptr;

static void*  g_dInput  = nullptr;
static void*  g_dOutput = nullptr;
static size_t g_frameByteSize = 1920 * 1080 * 4 * sizeof(float); // Adjust to your model's input tensor size

// Helper: Dynamically load whatever TensorRT / CUDA DLLs the user has in the game folder or PATH
static bool load_Libraries() {
    const char* trtNames[]  = { "nvinfer_10.dll", "nvinfer.dll", "nvinfer_8.dll" };
    const char* cudaNames[] = { "cudart64_12.dll", "cudart64_110.dll", "cudart.dll" };

    for (const char* name : trtNames) {
        g_hTrt = LoadLibraryA(name);
        if (g_hTrt) break;
    }
    for (const char* name : cudaNames) {
        g_hCuda = LoadLibraryA(name);
        if (g_hCuda) break;
    }

    if (!g_hTrt || !g_hCuda) {
        OutputDebugStringA("[NeuralRR] ERROR: Could not find nvinfer or cudart DLLs in game directory/PATH!\n");
        return false;
    }

    fn_createInferRuntime    = (pfn_createInferRuntime_INTERNAL)GetProcAddress(g_hTrt, "createInferRuntime_INTERNAL");
    fn_cudaMalloc            = (pfn_cudaMalloc)GetProcAddress(g_hCuda, "cudaMalloc");
    fn_cudaFree              = (pfn_cudaFree)GetProcAddress(g_hCuda, "cudaFree");
    fn_cudaMemcpyAsync       = (pfn_cudaMemcpyAsync)GetProcAddress(g_hCuda, "cudaMemcpyAsync");
    fn_cudaStreamCreate      = (pfn_cudaStreamCreate)GetProcAddress(g_hCuda, "cudaStreamCreate");
    fn_cudaStreamDestroy     = (pfn_cudaStreamDestroy)GetProcAddress(g_hCuda, "cudaStreamDestroy");
    fn_cudaStreamSynchronize = (pfn_cudaStreamSynchronize)GetProcAddress(g_hCuda, "cudaStreamSynchronize");

    return (fn_createInferRuntime && fn_cudaMalloc && fn_cudaFree && fn_cudaStreamCreate);
}

// ============================================================================
// 3. RESHADE EVENT HOOKS (INIT, FAST PRESENT, DESTROY)
// ============================================================================
static void on_init_device(reshade::api::device* device) {
    if (!load_Libraries()) return;

    // 1. Create dedicated non-blocking CUDA Stream
    fn_cudaStreamCreate(&g_cudaStream);

    // 2. Initialize TensorRT Runtime via internal C export
    g_trtRuntime = static_cast<nvinfer1::IRuntime*>(fn_createInferRuntime(&g_logger, NV_TENSORRT_VERSION));
    if (!g_trtRuntime) {
        OutputDebugStringA("[NeuralRR] ERROR: Failed to create TensorRT Runtime.\n");
        return;
    }

    // 3. Load pre-compiled FP16 .engine file from the game directory
    std::ifstream file("neuralrr_fp16.engine", std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        OutputDebugStringA("[NeuralRR] WARNING: neuralrr_fp16.engine not found in game directory.\n");
        return;
    }

    size_t size = file.tellg();
    std::vector<char> engineData(size);
    file.seekg(0, std::ios::beg);
    file.read(engineData.data(), size);
    file.close();

    g_trtEngine = g_trtRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!g_trtEngine) {
        OutputDebugStringA("[NeuralRR] ERROR: Engine deserialization failed.\n");
        return;
    }

    g_trtContext = g_trtEngine->createExecutionContext();

    // 4. Pre-allocate persistent VRAM buffers (Never allocate inside on_present!)
    fn_cudaMalloc(&g_dInput, g_frameByteSize);
    fn_cudaMalloc(&g_dOutput, g_frameByteSize);

    // Bind tensor addresses once if static shape (TensorRT 8.5+ / 10.x API)
    if (g_trtEngine->getNbIOTensors() >= 2) {
        const char* inputName  = g_trtEngine->getIOTensorName(0);
        const char* outputName = g_trtEngine->getIOTensorName(1);
        g_trtContext->setTensorAddress(inputName, g_dInput);
        g_trtContext->setTensorAddress(outputName, g_dOutput);
    }

    OutputDebugStringA("[NeuralRR] SUCCESS: TensorRT Engine & Persistent VRAM Initialized!\n");
}

static void on_present(
    reshade::api::command_queue* queue,
    reshade::api::swapchain* swapchain,
    const reshade::api::rect* source_rect,
    const reshade::api::rect* dest_rect,
    uint32_t dirty_rect_count,
    const reshade::api::rect* dirty_rects)
{
    if (!g_trtContext || !g_cudaStream) return;

    // Zero-overhead async inference dispatch on dedicated CUDA stream
    g_trtContext->enqueueV3(g_cudaStream);
}

static void on_destroy_device(reshade::api::device* device) {
    if (fn_cudaStreamSynchronize && g_cudaStream) {
        fn_cudaStreamSynchronize(g_cudaStream);
    }

    delete g_trtContext; g_trtContext = nullptr;
    delete g_trtEngine;  g_trtEngine  = nullptr;
    delete g_trtRuntime; g_trtRuntime = nullptr;

    if (fn_cudaFree) {
        if (g_dInput)  { fn_cudaFree(g_dInput);  g_dInput  = nullptr; }
        if (g_dOutput) { fn_cudaFree(g_dOutput); g_dOutput = nullptr; }
    }
    if (fn_cudaStreamDestroy && g_cudaStream) {
        fn_cudaStreamDestroy(g_cudaStream);
        g_cudaStream = nullptr;
    }

    if (g_hTrt)  { FreeLibrary(g_hTrt);  g_hTrt  = nullptr; }
    if (g_hCuda) { FreeLibrary(g_hCuda); g_hCuda = nullptr; }
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID lpvReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            if (!reshade::register_addon(hModule)) return FALSE;
            reshade::register_event<reshade::addon_event::init_device>(&on_init_device);
            reshade::register_event<reshade::addon_event::destroy_device>(&on_destroy_device);
            reshade::register_event<reshade::addon_event::present>(&on_present);
            break;
        case DLL_PROCESS_DETACH:
            reshade::unregister_event<reshade::addon_event::init_device>(&on_init_device);
            reshade::unregister_event<reshade::addon_event::destroy_device>(&on_destroy_device);
            reshade::unregister_event<reshade::addon_event::present>(&on_present);
            reshade::unregister_addon(hModule);
            break;
    }
    return TRUE;
}
