#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <reshade.hpp>
#include <NvInfer.h>
#include <fstream>
#include <vector>
#include <cstdint>

// ============================================================================
// 1. DYNAMIC CUDA & TENSORRT FUNCTION POINTERS (ZERO .LIB DEPENDENCIES)
// ============================================================================
typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;
typedef struct cudaGraphicsResource* cudaGraphicsResource_t;
typedef struct cudaArray* cudaArray_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

typedef cudaError_t (*pfn_cudaMalloc)(void** devPtr, size_t size);
typedef cudaError_t (*pfn_cudaFree)(void* devPtr);
typedef cudaError_t (*pfn_cudaStreamCreate)(cudaStream_t* pStream);
typedef cudaError_t (*pfn_cudaStreamDestroy)(cudaStream_t stream);
typedef cudaError_t (*pfn_cudaStreamSynchronize)(cudaStream_t stream);

// D3D11 Interop & 2D Array Memory Functions
typedef cudaError_t (*pfn_cudaGraphicsD3D11RegisterResource)(cudaGraphicsResource_t* resource, ID3D11Resource* d3dResource, unsigned int flags);
typedef cudaError_t (*pfn_cudaGraphicsUnregisterResource)(cudaGraphicsResource_t resource);
typedef cudaError_t (*pfn_cudaGraphicsMapResources)(int count, cudaGraphicsResource_t* resources, cudaStream_t stream);
typedef cudaError_t (*pfn_cudaGraphicsUnmapResources)(int count, cudaGraphicsResource_t* resources, cudaStream_t stream);
typedef cudaError_t (*pfn_cudaGraphicsSubResourceGetMappedArray)(cudaArray_t* array, cudaGraphicsResource_t resource, unsigned int arrayIndex, unsigned int mipLevel);
typedef cudaError_t (*pfn_cudaMemcpy2DFromArrayAsync)(void* dst, size_t dpitch, cudaArray_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
typedef cudaError_t (*pfn_cudaMemcpy2DToArrayAsync)(cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
typedef void* (*pfn_createInferRuntime_INTERNAL)(void* logger, int32_t version);

static HMODULE g_hCuda = nullptr;
static HMODULE g_hTrt  = nullptr;

static pfn_cudaMalloc                            fn_cudaMalloc = nullptr;
static pfn_cudaFree                              fn_cudaFree = nullptr;
static pfn_cudaStreamCreate                      fn_cudaStreamCreate = nullptr;
static pfn_cudaStreamDestroy                     fn_cudaStreamDestroy = nullptr;
static pfn_cudaStreamSynchronize                 fn_cudaStreamSynchronize = nullptr;
static pfn_cudaGraphicsD3D11RegisterResource     fn_cudaRegisterD3D11 = nullptr;
static pfn_cudaGraphicsUnregisterResource        fn_cudaUnregister = nullptr;
static pfn_cudaGraphicsMapResources              fn_cudaMap = nullptr;
static pfn_cudaGraphicsUnmapResources            fn_cudaUnmap = nullptr;
static pfn_cudaGraphicsSubResourceGetMappedArray fn_cudaGetArray = nullptr;
static pfn_cudaMemcpy2DFromArrayAsync            fn_cudaMemcpyFromArray = nullptr;
static pfn_cudaMemcpy2DToArrayAsync              fn_cudaMemcpyToArray = nullptr;
static pfn_createInferRuntime_INTERNAL           fn_createInferRuntime = nullptr;

// ============================================================================
// 2. GLOBAL ENGINE & INTEROP STATE
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

static void*                  g_dInput  = nullptr;
static void*                  g_dOutput = nullptr;
static reshade::api::resource g_texBridgeIn  = { 0 };
static reshade::api::resource g_texBridgeOut = { 0 };
static cudaGraphicsResource_t g_cudaResIn    = nullptr;
static cudaGraphicsResource_t g_cudaResOut   = nullptr;
static uint32_t               g_width = 0, g_height = 0;

static bool load_libraries() {
    // Added nvinfer_11.dll (for TensorRT 11.4) and cudart64_13.dll (for CUDA 13.4)
    const char* trtNames[]  = { "nvinfer_11.dll", "nvinfer_10.dll", "nvinfer.dll", "nvinfer_8.dll" };
    const char* cudaNames[] = { "cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart.dll" };

    for (const char* n : trtNames)  { if ((g_hTrt  = LoadLibraryA(n))) break; }
    for (const char* n : cudaNames) { if ((g_hCuda = LoadLibraryA(n))) break; }
    if (!g_hTrt || !g_hCuda) {
        OutputDebugStringA("[NeuralRR] ERROR: Missing nvinfer or cudart DLLs!\n");
        return false;
    }

    fn_createInferRuntime  = (pfn_createInferRuntime_INTERNAL)GetProcAddress(g_hTrt, "createInferRuntime_INTERNAL");
    fn_cudaMalloc          = (pfn_cudaMalloc)GetProcAddress(g_hCuda, "cudaMalloc");
    fn_cudaFree            = (pfn_cudaFree)GetProcAddress(g_hCuda, "cudaFree");
    fn_cudaStreamCreate    = (pfn_cudaStreamCreate)GetProcAddress(g_hCuda, "cudaStreamCreate");
    fn_cudaStreamDestroy   = (pfn_cudaStreamDestroy)GetProcAddress(g_hCuda, "cudaStreamDestroy");
    fn_cudaStreamSynchronize = (pfn_cudaStreamSynchronize)GetProcAddress(g_hCuda, "cudaStreamSynchronize");
    fn_cudaRegisterD3D11   = (pfn_cudaGraphicsD3D11RegisterResource)GetProcAddress(g_hCuda, "cudaGraphicsD3D11RegisterResource");
    fn_cudaUnregister      = (pfn_cudaGraphicsUnregisterResource)GetProcAddress(g_hCuda, "cudaGraphicsUnregisterResource");
    fn_cudaMap             = (pfn_cudaGraphicsMapResources)GetProcAddress(g_hCuda, "cudaGraphicsMapResources");
    fn_cudaUnmap           = (pfn_cudaGraphicsUnmapResources)GetProcAddress(g_hCuda, "cudaGraphicsUnmapResources");
    fn_cudaGetArray        = (pfn_cudaGraphicsSubResourceGetMappedArray)GetProcAddress(g_hCuda, "cudaGraphicsSubResourceGetMappedArray");
    fn_cudaMemcpyFromArray = (pfn_cudaMemcpy2DFromArrayAsync)GetProcAddress(g_hCuda, "cudaMemcpy2DFromArrayAsync");
    fn_cudaMemcpyToArray   = (pfn_cudaMemcpy2DToArrayAsync)GetProcAddress(g_hCuda, "cudaMemcpy2DToArrayAsync");

    return (fn_createInferRuntime && fn_cudaMalloc && fn_cudaRegisterD3D11);
}

// ============================================================================
// 3. RESHADE LIFECYCLE HOOKS
// ============================================================================
static void on_init_device(reshade::api::device* device) {
    if (!load_libraries()) return;
    fn_cudaStreamCreate(&g_cudaStream);

    g_trtRuntime = static_cast<nvinfer1::IRuntime*>(fn_createInferRuntime(&g_logger, NV_TENSORRT_VERSION));
    if (!g_trtRuntime) return;

    std::ifstream file("neuralrr_fp16.engine", std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        OutputDebugStringA("[NeuralRR] ERROR: neuralrr_fp16.engine not found in game folder!\n");
        return;
    }
    size_t size = file.tellg();
    std::vector<char> engineData(size);
    file.seekg(0, std::ios::beg);
    file.read(engineData.data(), size);
    file.close();

    g_trtEngine = g_trtRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!g_trtEngine) return;
    g_trtContext = g_trtEngine->createExecutionContext();
    OutputDebugStringA("[NeuralRR] SUCCESS: TensorRT Engine Loaded.\n");
}

static void on_init_swapchain(reshade::api::swapchain* swapchain, bool resize) {
    if (!g_trtContext || !fn_cudaRegisterD3D11) return;
    reshade::api::device* dev = swapchain->get_device();
    if (dev->get_api() != reshade::api::device_api::d3d11) return;

    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
    reshade::api::resource_desc desc = dev->get_resource_desc(backbuffer);
    g_width  = desc.texture.width;
    g_height = desc.texture.height;

    // Inside on_init_swapchain:
// 1x7xHxW FP16 = 7 * 2 bytes = 14 bytes/pixel. Allocate 16 bytes/pixel so it never overflows!
size_t inputBytes  = (size_t)g_width * g_height * 16;
size_t outputBytes = (size_t)g_width * g_height * 8;
fn_cudaMalloc(&g_dInput, inputBytes);
fn_cudaMalloc(&g_dOutput, outputBytes);

    if (g_trtEngine->getNbIOTensors() >= 2) {
        g_trtContext->setTensorAddress(g_trtEngine->getIOTensorName(0), g_dInput);
        g_trtContext->setTensorAddress(g_trtEngine->getIOTensorName(1), g_dOutput);
    }

    // Create two D3D11 staging textures in GPU VRAM for zero-copy CUDA bridging
    reshade::api::resource_desc bridgeDesc = desc;
    bridgeDesc.usage = reshade::api::resource_usage::copy_dest | reshade::api::resource_usage::copy_source | reshade::api::resource_usage::shader_resource;

    if (dev->create_resource(bridgeDesc, nullptr, reshade::api::resource_usage::general, &g_texBridgeIn) &&
        dev->create_resource(bridgeDesc, nullptr, reshade::api::resource_usage::general, &g_texBridgeOut))
    {
        ID3D11Resource* d3dIn  = reinterpret_cast<ID3D11Resource*>(g_texBridgeIn.handle);
        ID3D11Resource* d3dOut = reinterpret_cast<ID3D11Resource*>(g_texBridgeOut.handle);
        fn_cudaRegisterD3D11(&g_cudaResIn,  d3dIn,  0);
        fn_cudaRegisterD3D11(&g_cudaResOut, d3dOut, 0);
        OutputDebugStringA("[NeuralRR] SUCCESS: D3D11-CUDA Zero-Copy Interop Registered!\n");
    }
}

static void on_present(
    reshade::api::command_queue* queue,
    reshade::api::swapchain* swapchain,
    const reshade::api::rect* source_rect,
    const reshade::api::rect* dest_rect,
    uint32_t dirty_rect_count,
    const reshade::api::rect* dirty_rects)
{
    if (!g_trtContext || !g_cudaResIn || !g_cudaResOut) return;

    reshade::api::command_list* cmd = queue->get_immediate_command_list();
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();

    // 1. Copy Game Backbuffer -> D3D11 Input Bridge Texture
    cmd->copy_resource(backbuffer, g_texBridgeIn);
    queue->flush_immediate_command_list();

    // 2. Map D3D11 Textures into CUDA Space (Zero CPU involvement)
    cudaGraphicsResource_t resources[2] = { g_cudaResIn, g_cudaResOut };
    fn_cudaMap(2, resources, g_cudaStream);

    cudaArray_t arrIn = nullptr, arrOut = nullptr;
    fn_cudaGetArray(&arrIn,  g_cudaResIn,  0, 0);
    fn_cudaGetArray(&arrOut, g_cudaResOut, 0, 0);

    size_t pitch = (size_t)g_width * 4; // 4 bytes per pixel (RGBA8)
    fn_cudaMemcpyFromArray(g_dInput, pitch, arrIn, 0, 0, pitch, g_height, cudaMemcpyDeviceToDevice, g_cudaStream);

    // 3. Execute TensorRT Neural Pass
    g_trtContext->enqueueV3(g_cudaStream);

    // 4. Copy TensorRT Output -> D3D11 Output Bridge Texture & Unmap
    fn_cudaMemcpyToArray(arrOut, 0, 0, g_dOutput, pitch, pitch, g_height, cudaMemcpyDeviceToDevice, g_cudaStream);
    fn_cudaUnmap(2, resources, g_cudaStream);
    fn_cudaStreamSynchronize(g_cudaStream);

    // 5. Copy Processed Bridge Texture -> Game Backbuffer
    cmd->copy_resource(g_texBridgeOut, backbuffer);
}

static void on_destroy_swapchain(reshade::api::swapchain* swapchain, bool resize) {
    reshade::api::device* dev = swapchain->get_device();
    if (fn_cudaUnregister) {
        if (g_cudaResIn)  { fn_cudaUnregister(g_cudaResIn);  g_cudaResIn  = nullptr; }
        if (g_cudaResOut) { fn_cudaUnregister(g_cudaResOut); g_cudaResOut = nullptr; }
    }
    if (g_texBridgeIn.handle)  { dev->destroy_resource(g_texBridgeIn);  g_texBridgeIn  = { 0 }; }
    if (g_texBridgeOut.handle) { dev->destroy_resource(g_texBridgeOut); g_texBridgeOut = { 0 }; }
    if (fn_cudaFree) {
        if (g_dInput)  { fn_cudaFree(g_dInput);  g_dInput  = nullptr; }
        if (g_dOutput) { fn_cudaFree(g_dOutput); g_dOutput = nullptr; }
    }
}

static void on_destroy_device(reshade::api::device* device) {
    delete g_trtContext; g_trtContext = nullptr;
    delete g_trtEngine;  g_trtEngine  = nullptr;
    delete g_trtRuntime; g_trtRuntime = nullptr;
    if (fn_cudaStreamDestroy && g_cudaStream) { fn_cudaStreamDestroy(g_cudaStream); g_cudaStream = nullptr; }
    if (g_hTrt)  { FreeLibrary(g_hTrt);  g_hTrt  = nullptr; }
    if (g_hCuda) { FreeLibrary(g_hCuda); g_hCuda = nullptr; }
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID lpvReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            if (!reshade::register_addon(hModule)) return FALSE;
            reshade::register_event<reshade::addon_event::init_device>(&on_init_device);
            reshade::register_event<reshade::addon_event::init_swapchain>(&on_init_swapchain);
            reshade::register_event<reshade::addon_event::present>(&on_present);
            reshade::register_event<reshade::addon_event::destroy_swapchain>(&on_destroy_swapchain);
            reshade::register_event<reshade::addon_event::destroy_device>(&on_destroy_device);
            break;
        case DLL_PROCESS_DETACH:
            reshade::unregister_addon(hModule);
            break;
    }
    return TRUE;
}
