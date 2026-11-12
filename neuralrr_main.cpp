#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <reshade.hpp>
#include <NvInfer.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

// ============================================================================
// 1. DYNAMIC CUDA & TENSORRT FUNCTION POINTERS
// ============================================================================
typedef int cudaError_t;
typedef int CUresult;
typedef struct CUstream_st* cudaStream_t;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;

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
typedef cudaError_t (*pfn_cudaMemcpyAsync)(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
typedef cudaError_t (*pfn_cudaMemcpy2DAsync)(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream);
typedef void* (*pfn_createInferRuntime_INTERNAL)(void* logger, int32_t version);

typedef CUresult (*pfn_cuInit)(unsigned int Flags);
typedef CUresult (*pfn_cuModuleLoadData)(CUmodule* module, const void* image);
typedef CUresult (*pfn_cuModuleGetFunction)(CUfunction* hfunc, CUmodule hmod, const char* name);
typedef CUresult (*pfn_cuLaunchKernel)(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, cudaStream_t hStream, void** kernelParams, void** extra);

typedef struct _nvrtcProgram* nvrtcProgram;
typedef int nvrtcResult;
typedef nvrtcResult (*pfn_nvrtcCreateProgram)(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames);
typedef nvrtcResult (*pfn_nvrtcCompileProgram)(nvrtcProgram prog, int numOptions, const char* const* options);
typedef nvrtcResult (*pfn_nvrtcGetPTXSize)(nvrtcProgram prog, size_t* ptxSizeRet);
typedef nvrtcResult (*pfn_nvrtcGetPTX)(nvrtcProgram prog, char* ptx);
typedef nvrtcResult (*pfn_nvrtcDestroyProgram)(nvrtcProgram* prog);

static HMODULE g_hCuda   = nullptr;
static HMODULE g_hTrt    = nullptr;
static HMODULE g_hNvCuda = nullptr;
static HMODULE g_hNvrtc  = nullptr;

static pfn_cudaMalloc                  fn_cudaMalloc = nullptr;
static pfn_cudaFree                    fn_cudaFree = nullptr;
static pfn_cudaStreamCreate            fn_cudaStreamCreate = nullptr;
static pfn_cudaStreamDestroy           fn_cudaStreamDestroy = nullptr;
static pfn_cudaStreamSynchronize       fn_cudaStreamSynchronize = nullptr;
static pfn_cudaMemcpyAsync             fn_cudaMemcpyAsync = nullptr;
static pfn_cudaMemcpy2DAsync           fn_cudaMemcpy2DAsync = nullptr;
static pfn_createInferRuntime_INTERNAL fn_createInferRuntime = nullptr;

static pfn_cuInit              fn_cuInit = nullptr;
static pfn_cuModuleLoadData    fn_cuModuleLoadData = nullptr;
static pfn_cuModuleGetFunction fn_cuModuleGetFunction = nullptr;
static pfn_cuLaunchKernel      fn_cuLaunchKernel = nullptr;

static CUmodule   g_cuMod    = nullptr;
static CUfunction g_fnPack   = nullptr;
static CUfunction g_fnUnpack = nullptr;

// ============================================================================
// 2. GPU FORMAT KERNELS (BGRA8/RGBA8 <-> 1x7x1080x1920 FP16)
// ============================================================================
static const char* g_cudaKernelSrc = R"(
extern "C" __device__ unsigned short float_to_fp16(float f) {
    unsigned short ret;
    asm("{ cvt.rn.f16.f32 %0, %1; }" : "=h"(ret) : "f"(f));
    return ret;
}
extern "C" __device__ float fp16_to_float(unsigned short h) {
    float ret;
    asm("{ cvt.f32.f16 %0, %1; }" : "=f"(ret) : "h"(h));
    return ret;
}

extern "C" __global__ void pack_sdr2hdr_gbuffer(
    const unsigned char* __restrict__ rgba_in,
    unsigned short* __restrict__ gbuf_out,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    int rgba_idx = idx * 4;
    float r = (float)rgba_in[rgba_idx + 0] * (1.0f / 255.0f);
    float g = (float)rgba_in[rgba_idx + 1] * (1.0f / 255.0f);
    float b = (float)rgba_in[rgba_idx + 2] * (1.0f / 255.0f);
    float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;

    gbuf_out[0 * total_pixels + idx] = float_to_fp16(r);
    gbuf_out[1 * total_pixels + idx] = float_to_fp16(g);
    gbuf_out[2 * total_pixels + idx] = float_to_fp16(b);
    gbuf_out[3 * total_pixels + idx] = float_to_fp16(r * r);
    gbuf_out[4 * total_pixels + idx] = float_to_fp16(g * g);
    gbuf_out[5 * total_pixels + idx] = float_to_fp16(b * b);
    gbuf_out[6 * total_pixels + idx] = float_to_fp16(luma);
}

extern "C" __global__ void unpack_hdr_to_rgba(
    const unsigned short* __restrict__ hdr_in,
    unsigned char* __restrict__ rgba_out,
    int total_pixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pixels) return;

    float r = fp16_to_float(hdr_in[0 * total_pixels + idx]);
    float g = fp16_to_float(hdr_in[1 * total_pixels + idx]);
    float b = fp16_to_float(hdr_in[2 * total_pixels + idx]);

    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);

    int rgba_idx = idx * 4;
    rgba_out[rgba_idx + 0] = (unsigned char)(r * 255.0f + 0.5f);
    rgba_out[rgba_idx + 1] = (unsigned char)(g * 255.0f + 0.5f);
    rgba_out[rgba_idx + 2] = (unsigned char)(b * 255.0f + 0.5f);
    rgba_out[rgba_idx + 3] = 255;
}
)";

static inline uint16_t cpu_f32_to_f16(float f) {
    uint32_t x = *(uint32_t*)&f;
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | mant);
}
static inline float cpu_f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    if (exp == 0) return 0.0f;
    if (exp == 31) exp = 255; else exp = exp - 15 + 127;
    uint32_t x = sign | (exp << 23) | (mant << 13);
    return *(float*)&x;
}

// ============================================================================
// 3. GLOBAL ENGINE & DIRECTX 12 BUFFER STATE
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

static void*                g_dRgbaRaw = nullptr;
static void*                g_dInput   = nullptr;
static void*                g_dOutput  = nullptr;
static std::vector<uint16_t> g_cpuGbuf;
static std::vector<uint16_t> g_cpuOut;

// Native D3D12 Readback & Upload Buffers (Since D3D12 forbids TEXTURE_2D on CPU heaps!)
static ID3D12Resource*                  g_d3d12ReadbackBuf = nullptr;
static ID3D12Resource*                  g_d3d12UploadBuf   = nullptr;
static D3D12_PLACED_SUBRESOURCE_FOOTPRINT g_d3d12Footprint = {};
static uint8_t*                         g_mappedReadPtr    = nullptr;
static uint8_t*                         g_mappedWritePtr   = nullptr;

// Fallback ReShade textures for D3D11
static reshade::api::resource g_texRead  = { 0 };
static reshade::api::resource g_texWrite = { 0 };
static bool                   g_isD3D12  = false;

static void try_compile_gpu_kernels() {
    g_hNvCuda = LoadLibraryA("nvcuda.dll");
    const char* nvrtcNames[] = { "nvrtc64_130_0.dll", "nvrtc64_120_0.dll", "nvrtc64_112_0.dll", "nvrtc.dll" };
    for (const char* n : nvrtcNames) { if ((g_hNvrtc = LoadLibraryA(n))) break; }
    if (!g_hNvCuda || !g_hNvrtc) return;

    fn_cuInit              = (pfn_cuInit)GetProcAddress(g_hNvCuda, "cuInit");
    fn_cuModuleLoadData    = (pfn_cuModuleLoadData)GetProcAddress(g_hNvCuda, "cuModuleLoadData");
    fn_cuModuleGetFunction = (pfn_cuModuleGetFunction)GetProcAddress(g_hNvCuda, "cuModuleGetFunction");
    fn_cuLaunchKernel      = (pfn_cuLaunchKernel)GetProcAddress(g_hNvCuda, "cuLaunchKernel");

    auto fnCreate  = (pfn_nvrtcCreateProgram)GetProcAddress(g_hNvrtc, "nvrtcCreateProgram");
    auto fnCompile = (pfn_nvrtcCompileProgram)GetProcAddress(g_hNvrtc, "nvrtcCompileProgram");
    auto fnSize    = (pfn_nvrtcGetPTXSize)GetProcAddress(g_hNvrtc, "nvrtcGetPTXSize");
    auto fnPtx     = (pfn_nvrtcGetPTX)GetProcAddress(g_hNvrtc, "nvrtcGetPTX");
    auto fnDestroy = (pfn_nvrtcDestroyProgram)GetProcAddress(g_hNvrtc, "nvrtcDestroyProgram");

    if (!fn_cuModuleLoadData || !fnCreate) return;
    fn_cuInit(0);

    nvrtcProgram prog = nullptr;
    if (fnCreate(&prog, g_cudaKernelSrc, "sdr2hdr.cu", 0, nullptr, nullptr) == 0) {
        const char* opts[] = { "--gpu-architecture=compute_75" };
        if (fnCompile(prog, 1, opts) == 0) {
            size_t ptxSize = 0;
            fnSize(prog, &ptxSize);
            std::vector<char> ptx(ptxSize);
            fnPtx(prog, ptx.data());
            if (fn_cuModuleLoadData(&g_cuMod, ptx.data()) == 0) {
                fn_cuModuleGetFunction(&g_fnPack,   g_cuMod, "pack_sdr2hdr_gbuffer");
                fn_cuModuleGetFunction(&g_fnUnpack, g_cuMod, "unpack_hdr_to_rgba");
            }
        }
        fnDestroy(&prog);
    }
}

static bool load_libraries() {
    const char* trtNames[]  = { "nvinfer_11.dll", "nvinfer_10.dll", "nvinfer.dll", "nvinfer_8.dll" };
    const char* cudaNames[] = { "cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart.dll" };

    for (const char* n : trtNames)  { if ((g_hTrt  = LoadLibraryA(n))) break; }
    for (const char* n : cudaNames) { if ((g_hCuda = LoadLibraryA(n))) break; }
    if (!g_hTrt || !g_hCuda) return false;

    fn_createInferRuntime    = (pfn_createInferRuntime_INTERNAL)GetProcAddress(g_hTrt, "createInferRuntime_INTERNAL");
    fn_cudaMalloc            = (pfn_cudaMalloc)GetProcAddress(g_hCuda, "cudaMalloc");
    fn_cudaFree              = (pfn_cudaFree)GetProcAddress(g_hCuda, "cudaFree");
    fn_cudaStreamCreate      = (pfn_cudaStreamCreate)GetProcAddress(g_hCuda, "cudaStreamCreate");
    fn_cudaStreamDestroy     = (pfn_cudaStreamDestroy)GetProcAddress(g_hCuda, "cudaStreamDestroy");
    fn_cudaStreamSynchronize = (pfn_cudaStreamSynchronize)GetProcAddress(g_hCuda, "cudaStreamSynchronize");
    fn_cudaMemcpyAsync       = (pfn_cudaMemcpyAsync)GetProcAddress(g_hCuda, "cudaMemcpyAsync");
    fn_cudaMemcpy2DAsync     = (pfn_cudaMemcpy2DAsync)GetProcAddress(g_hCuda, "cudaMemcpy2DAsync");

    try_compile_gpu_kernels();
    return (fn_createInferRuntime && fn_cudaMalloc && fn_cudaStreamCreate);
}

static void on_init_device(reshade::api::device* device) {
    if (!load_libraries()) return;
    fn_cudaStreamCreate(&g_cudaStream);
    g_isD3D12 = (device->get_api() == reshade::api::device_api::d3d12);

    // Fix Bug 1: Probe exact TensorRT version (110400, NV_TENSORRT_VERSION, etc.) so TRT 11.4 never rejects header version!
    int32_t candidateVersions[] = { 110400, 110300, 110200, 110100, 110000, NV_TENSORRT_VERSION, 100300, 100000 };
    for (int32_t v : candidateVersions) {
        g_trtRuntime = static_cast<nvinfer1::IRuntime*>(fn_createInferRuntime(&g_logger, v));
        if (g_trtRuntime) break;
    }
    if (!g_trtRuntime) {
        for (int32_t v = 100000; v <= 120000; v += 100) {
            g_trtRuntime = static_cast<nvinfer1::IRuntime*>(fn_createInferRuntime(&g_logger, v));
            if (g_trtRuntime) break;
        }
    }
    if (!g_trtRuntime) return;

    std::ifstream file("neuralrr_fp16.engine", std::ios::binary | std::ios::ate);
    if (!file.is_open()) return;
    size_t size = file.tellg();
    std::vector<char> engineData(size);
    file.seekg(0, std::ios::beg);
    file.read(engineData.data(), size);
    file.close();

    g_trtEngine = g_trtRuntime->deserializeCudaEngine(engineData.data(), size);
    if (!g_trtEngine) return;
    g_trtContext = g_trtEngine->createExecutionContext();
}

static void on_init_swapchain(reshade::api::swapchain* swapchain, bool resize) {
    if (!g_trtContext) return;
    reshade::api::device* dev = swapchain->get_device();

    size_t pixels = (size_t)1920 * 1080;
    fn_cudaMalloc(&g_dRgbaRaw, pixels * 4);
    fn_cudaMalloc(&g_dInput,   pixels * 7 * sizeof(uint16_t));
    fn_cudaMalloc(&g_dOutput,  pixels * 3 * sizeof(uint16_t));

    if (!g_fnPack) {
        g_cpuGbuf.resize(pixels * 7);
        g_cpuOut.resize(pixels * 3);
    }

    if (g_trtEngine->getNbIOTensors() >= 2) {
        const char* inName  = g_trtEngine->getIOTensorName(0);
        const char* outName = g_trtEngine->getIOTensorName(1);
        
        nvinfer1::Dims inDims;
        inDims.nbDims = 4;
        inDims.d[0] = 1; inDims.d[1] = 7; inDims.d[2] = 1080; inDims.d[3] = 1920;
        g_trtContext->setInputShape(inName, inDims);

        g_trtContext->setTensorAddress(inName,  g_dInput);
        g_trtContext->setTensorAddress(outName, g_dOutput);
    }

    // Fix Bug 2: In DirectX 12, create native D3D12 Readback & Upload linear buffers (persistent mapped!)
    if (g_isD3D12) {
        ID3D12Device* d3d12Dev = reinterpret_cast<ID3D12Device*>(dev->get_native());
        ID3D12Resource* bbRes  = reinterpret_cast<ID3D12Resource*>(swapchain->get_current_back_buffer().handle);
        D3D12_RESOURCE_DESC bbDesc = bbRes->GetDesc();

        UINT64 totalBytes = 0;
        d3d12Dev->GetCopyableFootprints(&bbDesc, 0, 1, 0, &g_d3d12Footprint, nullptr, nullptr, &totalBytes);

        D3D12_HEAP_PROPERTIES readHeap = {};
        readHeap.Type = D3D12_HEAP_TYPE_READBACK;

        D3D12_HEAP_PROPERTIES writeHeap = {};
        writeHeap.Type = D3D12_HEAP_TYPE_UPLOAD;

        D3D12_RESOURCE_DESC bufDesc = {};
        bufDesc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufDesc.Width              = totalBytes;
        bufDesc.Height             = 1;
        bufDesc.DepthOrArraySize   = 1;
        bufDesc.MipLevels          = 1;
        bufDesc.Format             = DXGI_FORMAT_UNKNOWN;
        bufDesc.SampleDesc.Count   = 1;
        bufDesc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        if (SUCCEEDED(d3d12Dev->CreateCommittedResource(&readHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_d3d12ReadbackBuf)))) {
            g_d3d12ReadbackBuf->Map(0, nullptr, reinterpret_cast<void**>(&g_mappedReadPtr));
        }
        if (SUCCEEDED(d3d12Dev->CreateCommittedResource(&writeHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_d3d12UploadBuf)))) {
            g_d3d12UploadBuf->Map(0, nullptr, reinterpret_cast<void**>(&g_mappedWritePtr));
        }
    } else {
        reshade::api::resource backbuffer = swapchain->get_current_back_buffer();
        reshade::api::resource_desc desc = dev->get_resource_desc(backbuffer);

        reshade::api::resource_desc readDesc = desc;
        readDesc.heap  = reshade::api::memory_heap::gpu_to_cpu;
        readDesc.usage = reshade::api::resource_usage::copy_dest;

        reshade::api::resource_desc writeDesc = desc;
        writeDesc.heap  = reshade::api::memory_heap::cpu_to_gpu;
        writeDesc.usage = reshade::api::resource_usage::copy_source;

        dev->create_resource(readDesc,  nullptr, reshade::api::resource_usage::copy_dest,   &g_texRead);
        dev->create_resource(writeDesc, nullptr, reshade::api::resource_usage::copy_source, &g_texWrite);
    }
}

static void process_frame_pixels(const uint8_t* srcPtr, size_t srcPitch, uint8_t* dstPtr, size_t dstPitch) {
    int totalPixels = 1920 * 1080;
    size_t rowBytes = 1920 * 4;

    if (g_fnPack && g_fnUnpack && fn_cuLaunchKernel) {
        fn_cudaMemcpy2DAsync(g_dRgbaRaw, rowBytes, srcPtr, srcPitch, rowBytes, 1080, cudaMemcpyHostToDevice, g_cudaStream);

        int threads = 256;
        int blocks  = (totalPixels + threads - 1) / threads;
        void* packArgs[] = { &g_dRgbaRaw, &g_dInput, &totalPixels };
        fn_cuLaunchKernel(g_fnPack, blocks, 1, 1, threads, 1, 1, 0, g_cudaStream, packArgs, nullptr);

        g_trtContext->enqueueV3(g_cudaStream);

        void* unpackArgs[] = { &g_dOutput, &g_dRgbaRaw, &totalPixels };
        fn_cuLaunchKernel(g_fnUnpack, blocks, 1, 1, threads, 1, 1, 0, g_cudaStream, unpackArgs, nullptr);

        fn_cudaMemcpy2DAsync(dstPtr, dstPitch, g_dRgbaRaw, rowBytes, rowBytes, 1080, cudaMemcpyDeviceToHost, g_cudaStream);
        fn_cudaStreamSynchronize(g_cudaStream);
    } else {
        for (int y = 0; y < 1080; ++y) {
            const uint8_t* row = srcPtr + y * srcPitch;
            for (int x = 0; x < 1920; ++x) {
                int idx = y * 1920 + x;
                float r = row[x * 4 + 0] * (1.0f / 255.0f);
                float g = row[x * 4 + 1] * (1.0f / 255.0f);
                float b = row[x * 4 + 2] * (1.0f / 255.0f);
                float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;

                g_cpuGbuf[0 * totalPixels + idx] = cpu_f32_to_f16(r);
                g_cpuGbuf[1 * totalPixels + idx] = cpu_f32_to_f16(g);
                g_cpuGbuf[2 * totalPixels + idx] = cpu_f32_to_f16(b);
                g_cpuGbuf[3 * totalPixels + idx] = cpu_f32_to_f16(r * r);
                g_cpuGbuf[4 * totalPixels + idx] = cpu_f32_to_f16(g * g);
                g_cpuGbuf[5 * totalPixels + idx] = cpu_f32_to_f16(b * b);
                g_cpuGbuf[6 * totalPixels + idx] = cpu_f32_to_f16(luma);
            }
        }

        fn_cudaMemcpyAsync(g_dInput, g_cpuGbuf.data(), totalPixels * 7 * sizeof(uint16_t), cudaMemcpyHostToDevice, g_cudaStream);
        g_trtContext->enqueueV3(g_cudaStream);
        fn_cudaMemcpyAsync(g_cpuOut.data(), g_dOutput, totalPixels * 3 * sizeof(uint16_t), cudaMemcpyDeviceToHost, g_cudaStream);
        fn_cudaStreamSynchronize(g_cudaStream);

        for (int y = 0; y < 1080; ++y) {
            uint8_t* dstRow = dstPtr + y * dstPitch;
            for (int x = 0; x < 1920; ++x) {
                int idx = y * 1920 + x;
                float r = cpu_f16_to_f32(g_cpuOut[0 * totalPixels + idx]);
                float g = cpu_f16_to_f32(g_cpuOut[1 * totalPixels + idx]);
                float b = cpu_f16_to_f32(g_cpuOut[2 * totalPixels + idx]);
                r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
                g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
                b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
                dstRow[x * 4 + 0] = (uint8_t)(r * 255.0f);
                dstRow[x * 4 + 1] = (uint8_t)(g * 255.0f);
                dstRow[x * 4 + 2] = (uint8_t)(b * 255.0f);
                dstRow[x * 4 + 3] = 255;
            }
        }
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
    if (!g_trtContext) return;

    reshade::api::device* dev = swapchain->get_device();
    reshade::api::command_list* cmd = queue->get_immediate_command_list();
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();

    if (g_isD3D12 && g_d3d12ReadbackBuf && g_d3d12UploadBuf && g_mappedReadPtr && g_mappedWritePtr) {
        ID3D12GraphicsCommandList* d3d12Cmd = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native());
        ID3D12Resource* bbRes = reinterpret_cast<ID3D12Resource*>(backbuffer.handle);

        // 1. Transition D3D12 Backbuffer: PRESENT -> COPY_SOURCE
        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource   = bbRes;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        d3d12Cmd->ResourceBarrier(1, &barrier);

        // 2. Copy Backbuffer Texture -> D3D12 Readback Buffer
        D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
        srcLoc.pResource        = bbRes;
        srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
        dstLoc.pResource       = g_d3d12ReadbackBuf;
        dstLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dstLoc.PlacedFootprint = g_d3d12Footprint;

        d3d12Cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

        // Wait for D3D12 copy to finish
        queue->flush_immediate_command_list();
        queue->wait_idle();

        // 3. Process Frame Through TensorRT SDR2HDR Engine
        process_frame_pixels(g_mappedReadPtr, g_d3d12Footprint.Footprint.RowPitch, g_mappedWritePtr, g_d3d12Footprint.Footprint.RowPitch);

        // 4. Transition Backbuffer: COPY_SOURCE -> COPY_DEST
        cmd = queue->get_immediate_command_list();
        d3d12Cmd = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native());

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        d3d12Cmd->ResourceBarrier(1, &barrier);

        // 5. Copy Processed Upload Buffer -> D3D12 Backbuffer Texture
        D3D12_TEXTURE_COPY_LOCATION uploadSrc = {};
        uploadSrc.pResource       = g_d3d12UploadBuf;
        uploadSrc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        uploadSrc.PlacedFootprint = g_d3d12Footprint;

        D3D12_TEXTURE_COPY_LOCATION bbDst = {};
        bbDst.pResource        = bbRes;
        bbDst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        bbDst.SubresourceIndex = 0;

        d3d12Cmd->CopyTextureRegion(&bbDst, 0, 0, 0, &uploadSrc, nullptr);

        // 6. Transition Backbuffer: COPY_DEST -> PRESENT
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        d3d12Cmd->ResourceBarrier(1, &barrier);

        queue->flush_immediate_command_list();
    }
    else if (g_texRead.handle && g_texWrite.handle) {
        cmd->barrier(backbuffer, reshade::api::resource_usage::present, reshade::api::resource_usage::copy_source);
        cmd->copy_resource(backbuffer, g_texRead);
        cmd->barrier(backbuffer, reshade::api::resource_usage::copy_source, reshade::api::resource_usage::present);
        queue->flush_immediate_command_list();
        queue->wait_idle();

        reshade::api::subresource_data mapIn = {}, mapOut = {};
        if (dev->map_texture_region(g_texRead, 0, nullptr, reshade::api::map_access::read_only, &mapIn)) {
            if (dev->map_texture_region(g_texWrite, 0, nullptr, reshade::api::map_access::write_only, &mapOut)) {
                process_frame_pixels(static_cast<const uint8_t*>(mapIn.data), mapIn.row_pitch, static_cast<uint8_t*>(mapOut.data), mapOut.row_pitch);
                dev->unmap_texture_region(g_texWrite, 0);
            }
            dev->unmap_texture_region(g_texRead, 0);
        }

        cmd->barrier(backbuffer, reshade::api::resource_usage::present, reshade::api::resource_usage::copy_dest);
        cmd->copy_resource(g_texWrite, backbuffer);
        cmd->barrier(backbuffer, reshade::api::resource_usage::copy_dest, reshade::api::resource_usage::present);
        queue->flush_immediate_command_list();
    }
}

static void on_destroy_swapchain(reshade::api::swapchain* swapchain, bool resize) {
    reshade::api::device* dev = swapchain->get_device();
    if (g_d3d12ReadbackBuf) { g_d3d12ReadbackBuf->Unmap(0, nullptr); g_d3d12ReadbackBuf->Release(); g_d3d12ReadbackBuf = nullptr; g_mappedReadPtr = nullptr; }
    if (g_d3d12UploadBuf)   { g_d3d12UploadBuf->Unmap(0, nullptr);   g_d3d12UploadBuf->Release();   g_d3d12UploadBuf   = nullptr; g_mappedWritePtr = nullptr; }
    if (g_texRead.handle)   { dev->destroy_resource(g_texRead);  g_texRead  = { 0 }; }
    if (g_texWrite.handle)  { dev->destroy_resource(g_texWrite); g_texWrite = { 0 }; }
    if (fn_cudaFree) {
        if (g_dRgbaRaw) { fn_cudaFree(g_dRgbaRaw); g_dRgbaRaw = nullptr; }
        if (g_dInput)   { fn_cudaFree(g_dInput);   g_dInput   = nullptr; }
        if (g_dOutput)  { fn_cudaFree(g_dOutput);  g_dOutput  = nullptr; }
    }
}

static void on_destroy_device(reshade::api::device* device) {
    if (fn_cudaStreamSynchronize && g_cudaStream) fn_cudaStreamSynchronize(g_cudaStream);
    delete g_trtContext; g_trtContext = nullptr;
    delete g_trtEngine;  g_trtEngine  = nullptr;
    delete g_trtRuntime; g_trtRuntime = nullptr;
    if (fn_cudaStreamDestroy && g_cudaStream) { fn_cudaStreamDestroy(g_cudaStream); g_cudaStream = nullptr; }
    if (g_hTrt)    { FreeLibrary(g_hTrt);    g_hTrt    = nullptr; }
    if (g_hCuda)   { FreeLibrary(g_hCuda);   g_hCuda   = nullptr; }
    if (g_hNvCuda) { FreeLibrary(g_hNvCuda); g_hNvCuda = nullptr; }
    if (g_hNvrtc)  { FreeLibrary(g_hNvrtc);  g_hNvrtc  = nullptr; }
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
