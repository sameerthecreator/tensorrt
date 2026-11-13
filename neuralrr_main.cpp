#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <reshade.hpp>
#include <NvInfer.h>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

typedef int cudaError_t;
typedef struct CUstream_st* cudaStream_t;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3
};

typedef cudaError_t (*pfn_cudaMalloc)(void** devPtr, size_t size);
typedef cudaError_t (*pfn_cudaMemset)(void* devPtr, int value, size_t count);
typedef cudaError_t (*pfn_cudaFree)(void* devPtr);
typedef cudaError_t (*pfn_cudaStreamCreate)(cudaStream_t* pStream);
typedef cudaError_t (*pfn_cudaStreamDestroy)(cudaStream_t stream);
typedef cudaError_t (*pfn_cudaStreamSynchronize)(cudaStream_t stream);
typedef cudaError_t (*pfn_cudaMemcpyAsync)(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream);
typedef void* (*pfn_createInferRuntime_INTERNAL)(void* logger, int32_t version);

static HMODULE g_hCuda = nullptr;
static HMODULE g_hTrt  = nullptr;

static pfn_cudaMalloc                  fn_cudaMalloc = nullptr;
static pfn_cudaMemset                  fn_cudaMemset = nullptr;
static pfn_cudaFree                    fn_cudaFree = nullptr;
static pfn_cudaStreamCreate            fn_cudaStreamCreate = nullptr;
static pfn_cudaStreamDestroy           fn_cudaStreamDestroy = nullptr;
static pfn_cudaStreamSynchronize       fn_cudaStreamSynchronize = nullptr;
static pfn_cudaMemcpyAsync             fn_cudaMemcpyAsync = nullptr;
static pfn_createInferRuntime_INTERNAL fn_createInferRuntime = nullptr;

static float g_fp16ToFloat[65536];
static bool  g_tablesInitialized = false;

static uint16_t calc_f32_to_f16(float f) {
    if (!(f > 0.0f)) return 0;
    if (f >= 65504.0f) f = 65504.0f;
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static float calc_f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
        exp++; mant &= ~0x0400;
    } else if (exp == 31) {
        return 0.0f;
    }
    exp = exp - 15 + 127;
    uint32_t x = sign | (exp << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

static void init_lookup_tables() {
    if (g_tablesInitialized) return;
    for (int i = 0; i < 65536; ++i) {
        g_fp16ToFloat[i] = calc_f16_to_f32((uint16_t)i);
    }
    g_tablesInitialized = true;
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {}
} g_logger;

static nvinfer1::IRuntime*          g_trtRuntime = nullptr;
static nvinfer1::ICudaEngine*       g_trtEngine  = nullptr;
static nvinfer1::IExecutionContext* g_trtContext = nullptr;
static cudaStream_t                 g_cudaStream = nullptr;

static constexpr int ENGINE_W = 1920;
static constexpr int ENGINE_H = 1080;
static constexpr int ENGINE_PIXELS = ENGINE_W * ENGINE_H;

static void*                 g_dInput   = nullptr;
static void*                 g_dOutput  = nullptr;
static std::vector<uint16_t> g_cpuGbuf;
static std::vector<uint16_t> g_cpuOut;

static ID3D12Resource*                    g_d3d12ReadbackBuf = nullptr;
static ID3D12Resource*                    g_d3d12UploadBuf   = nullptr;
static D3D12_PLACED_SUBRESOURCE_FOOTPRINT g_d3d12Footprint   = {};
static uint8_t*                           g_mappedReadPtr    = nullptr;
static uint8_t*                           g_mappedWritePtr   = nullptr;

static bool        g_isD3D12   = false;
static uint32_t    g_bbWidth   = 1920;
static uint32_t    g_bbHeight  = 1080;
static DXGI_FORMAT g_dxgiFormat = DXGI_FORMAT_R8G8B8A8_UNORM;

// Reads exact Native DXGI_FORMAT from ID3D12Resource::GetDesc().Format!
static inline void read_dxgi_pixel(const uint8_t* row, uint32_t x, DXGI_FORMAT fmt, float& r, float& g, float& b, uint64_t& rawOrig) {
    if (fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(row + x * 8);
        rawOrig = *reinterpret_cast<const uint64_t*>(p);
        r = g_fp16ToFloat[p[0]];
        g = g_fp16ToFloat[p[1]];
        b = g_fp16ToFloat[p[2]];
    } else if (fmt == DXGI_FORMAT_R10G10B10A2_UNORM || fmt == DXGI_FORMAT_R10G10B10A2_TYPELESS || fmt == DXGI_FORMAT_R10G10B10A2_UINT) {
        uint32_t v = *reinterpret_cast<const uint32_t*>(row + x * 4);
        rawOrig = v;
        r = (float)((v >>  0) & 0x3FF) * (1.0f / 1023.0f);
        g = (float)((v >> 10) & 0x3FF) * (1.0f / 1023.0f);
        b = (float)((v >> 20) & 0x3FF) * (1.0f / 1023.0f);
    } else if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM || fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || fmt == DXGI_FORMAT_B8G8R8A8_TYPELESS ||
               fmt == DXGI_FORMAT_B8G8R8X8_UNORM || fmt == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB || fmt == DXGI_FORMAT_B8G8R8X8_TYPELESS) {
        const uint8_t* p = row + x * 4;
        rawOrig = *reinterpret_cast<const uint32_t*>(p);
        b = p[0] * (1.0f / 255.0f);
        g = p[1] * (1.0f / 255.0f);
        r = p[2] * (1.0f / 255.0f);
    } else {
        const uint8_t* p = row + x * 4;
        rawOrig = *reinterpret_cast<const uint32_t*>(p);
        r = p[0] * (1.0f / 255.0f);
        g = p[1] * (1.0f / 255.0f);
        b = p[2] * (1.0f / 255.0f);
    }
    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);
}

static inline void write_dxgi_pixel(uint8_t* row, uint32_t x, DXGI_FORMAT fmt, float r, float g, float b, uint64_t rawOrig) {
    r = r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r);
    g = g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g);
    b = b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b);

    if (fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
        uint16_t* p = reinterpret_cast<uint16_t*>(row + x * 8);
        p[0] = calc_f32_to_f16(r);
        p[1] = calc_f32_to_f16(g);
        p[2] = calc_f32_to_f16(b);
        p[3] = (uint16_t)((rawOrig >> 48) & 0xFFFF);
    } else if (fmt == DXGI_FORMAT_R10G10B10A2_UNORM || fmt == DXGI_FORMAT_R10G10B10A2_TYPELESS || fmt == DXGI_FORMAT_R10G10B10A2_UINT) {
        uint32_t ir = (uint32_t)(r * 1023.0f + 0.5f) & 0x3FF;
        uint32_t ig = (uint32_t)(g * 1023.0f + 0.5f) & 0x3FF;
        uint32_t ib = (uint32_t)(b * 1023.0f + 0.5f) & 0x3FF;
        uint32_t ia = (uint32_t)(rawOrig & 0xC0000000u);
        *reinterpret_cast<uint32_t*>(row + x * 4) = ir | (ig << 10) | (ib << 20) | ia;
    } else if (fmt == DXGI_FORMAT_B8G8R8A8_UNORM || fmt == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || fmt == DXGI_FORMAT_B8G8R8A8_TYPELESS ||
               fmt == DXGI_FORMAT_B8G8R8X8_UNORM || fmt == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB || fmt == DXGI_FORMAT_B8G8R8X8_TYPELESS) {
        uint8_t* p = row + x * 4;
        p[0] = (uint8_t)(b * 255.0f + 0.5f);
        p[1] = (uint8_t)(g * 255.0f + 0.5f);
        p[2] = (uint8_t)(r * 255.0f + 0.5f);
        p[3] = (uint8_t)((rawOrig >> 24) & 0xFF);
    } else {
        uint8_t* p = row + x * 4;
        p[0] = (uint8_t)(r * 255.0f + 0.5f);
        p[1] = (uint8_t)(g * 255.0f + 0.5f);
        p[2] = (uint8_t)(b * 255.0f + 0.5f);
        p[3] = (uint8_t)((rawOrig >> 24) & 0xFF);
    }
}

static bool load_libraries() {
    init_lookup_tables();
    const char* trtNames[]  = { "nvinfer_11.dll", "nvinfer_10.dll", "nvinfer.dll", "nvinfer_8.dll" };
    const char* cudaNames[] = { "cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart.dll" };

    for (const char* n : trtNames)  { if ((g_hTrt  = LoadLibraryA(n))) break; }
    for (const char* n : cudaNames) { if ((g_hCuda = LoadLibraryA(n))) break; }
    if (!g_hTrt || !g_hCuda) return false;

    fn_createInferRuntime    = (pfn_createInferRuntime_INTERNAL)GetProcAddress(g_hTrt, "createInferRuntime_INTERNAL");
    fn_cudaMalloc            = (pfn_cudaMalloc)GetProcAddress(g_hCuda, "cudaMalloc");
    fn_cudaMemset            = (pfn_cudaMemset)GetProcAddress(g_hCuda, "cudaMemset");
    fn_cudaFree              = (pfn_cudaFree)GetProcAddress(g_hCuda, "cudaFree");
    fn_cudaStreamCreate      = (pfn_cudaStreamCreate)GetProcAddress(g_hCuda, "cudaStreamCreate");
    fn_cudaStreamDestroy     = (pfn_cudaStreamDestroy)GetProcAddress(g_hCuda, "cudaStreamDestroy");
    fn_cudaStreamSynchronize = (pfn_cudaStreamSynchronize)GetProcAddress(g_hCuda, "cudaStreamSynchronize");
    fn_cudaMemcpyAsync       = (pfn_cudaMemcpyAsync)GetProcAddress(g_hCuda, "cudaMemcpyAsync");

    return (fn_createInferRuntime && fn_cudaMalloc && fn_cudaStreamCreate && fn_cudaMemcpyAsync);
}

static void on_init_device(reshade::api::device* device) {
    if (!load_libraries()) return;
    fn_cudaStreamCreate(&g_cudaStream);
    g_isD3D12 = (device->get_api() == reshade::api::device_api::d3d12);

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
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();

    fn_cudaMalloc(&g_dInput,  (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t));
    fn_cudaMalloc(&g_dOutput, (size_t)ENGINE_PIXELS * 3 * sizeof(uint16_t));
    if (fn_cudaMemset) {
        fn_cudaMemset(g_dInput,  0, (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t));
        fn_cudaMemset(g_dOutput, 0, (size_t)ENGINE_PIXELS * 3 * sizeof(uint16_t));
    }

    g_cpuGbuf.assign((size_t)ENGINE_PIXELS * 7, 0);
    g_cpuOut.assign((size_t)ENGINE_PIXELS * 3, 0);

    uint16_t half05 = calc_f32_to_f16(0.5f);
    uint16_t half10 = calc_f32_to_f16(1.0f);
    for (int i = 0; i < ENGINE_PIXELS; ++i) {
        g_cpuGbuf[3 * ENGINE_PIXELS + i] = half05;
        g_cpuGbuf[4 * ENGINE_PIXELS + i] = half05;
        g_cpuGbuf[5 * ENGINE_PIXELS + i] = half10;
    }

    nvinfer1::Dims inDims;
    inDims.nbDims = 4;
    inDims.d[0] = 1; inDims.d[1] = 7; inDims.d[2] = 1080; inDims.d[3] = 1920;
    g_trtContext->setInputShape("g_buffer", inDims);
    g_trtContext->setTensorAddress("g_buffer", g_dInput);
    g_trtContext->setTensorAddress("rendered_frame", g_dOutput);

    if (g_isD3D12) {
        ID3D12Device* d3d12Dev = reinterpret_cast<ID3D12Device*>(dev->get_native());
        ID3D12Resource* bbRes  = reinterpret_cast<ID3D12Resource*>(backbuffer.handle);
        D3D12_RESOURCE_DESC bbDesc = bbRes->GetDesc();

        g_bbWidth    = (uint32_t)bbDesc.Width;
        g_bbHeight   = (uint32_t)bbDesc.Height;
        g_dxgiFormat = bbDesc.Format; // Exact native DXGI_FORMAT!

        UINT64 totalBytes = 0;
        d3d12Dev->GetCopyableFootprints(&bbDesc, 0, 1, 0, &g_d3d12Footprint, nullptr, nullptr, &totalBytes);

        D3D12_HEAP_PROPERTIES readHeap  = { D3D12_HEAP_TYPE_READBACK };
        D3D12_HEAP_PROPERTIES writeHeap = { D3D12_HEAP_TYPE_UPLOAD };

        D3D12_RESOURCE_DESC bufDesc = {};
        bufDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        bufDesc.Width            = totalBytes;
        bufDesc.Height           = 1;
        bufDesc.DepthOrArraySize = 1;
        bufDesc.MipLevels        = 1;
        bufDesc.Format           = DXGI_FORMAT_UNKNOWN;
        bufDesc.SampleDesc.Count = 1;
        bufDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        if (SUCCEEDED(d3d12Dev->CreateCommittedResource(&readHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&g_d3d12ReadbackBuf)))) {
            g_d3d12ReadbackBuf->Map(0, nullptr, reinterpret_cast<void**>(&g_mappedReadPtr));
        }
        if (SUCCEEDED(d3d12Dev->CreateCommittedResource(&writeHeap, D3D12_HEAP_FLAG_NONE, &bufDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&g_d3d12UploadBuf)))) {
            g_d3d12UploadBuf->Map(0, nullptr, reinterpret_cast<void**>(&g_mappedWritePtr));
        }
    }
}

static void process_frame_pixels(const uint8_t* srcPtr, size_t srcPitch, uint8_t* dstPtr, size_t dstPitch) {
    uint16_t* ch0 = &g_cpuGbuf[0 * ENGINE_PIXELS];
    uint16_t* ch1 = &g_cpuGbuf[1 * ENGINE_PIXELS];
    uint16_t* ch2 = &g_cpuGbuf[2 * ENGINE_PIXELS];
    uint16_t* ch6 = &g_cpuGbuf[6 * ENGINE_PIXELS];

    for (int ey = 0; ey < ENGINE_H; ++ey) {
        uint32_t sy = (ey * g_bbHeight) / ENGINE_H;
        const uint8_t* row = srcPtr + sy * srcPitch;
        int rowBase = ey * ENGINE_W;

        for (int ex = 0; ex < ENGINE_W; ++ex) {
            uint32_t sx = (ex * g_bbWidth) / ENGINE_W;
            float r, g, b;
            uint64_t rawOrig;
            read_dxgi_pixel(row, sx, g_dxgiFormat, r, g, b, rawOrig);
            float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;

            int idx = rowBase + ex;
            ch0[idx] = calc_f32_to_f16(r);
            ch1[idx] = calc_f32_to_f16(g);
            ch2[idx] = calc_f32_to_f16(b);
            ch6[idx] = calc_f32_to_f16(luma);
        }
    }

    fn_cudaMemcpyAsync(g_dInput, g_cpuGbuf.data(), (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t), cudaMemcpyHostToDevice, g_cudaStream);
    g_trtContext->enqueueV3(g_cudaStream);
    fn_cudaMemcpyAsync(g_cpuOut.data(), g_dOutput, (size_t)ENGINE_PIXELS * 3 * sizeof(uint16_t), cudaMemcpyDeviceToHost, g_cudaStream);
    fn_cudaStreamSynchronize(g_cudaStream);

    const uint16_t* outR = &g_cpuOut[0 * ENGINE_PIXELS];
    const uint16_t* outG = &g_cpuOut[1 * ENGINE_PIXELS];
    const uint16_t* outB = &g_cpuOut[2 * ENGINE_PIXELS];

    for (uint32_t y = 0; y < g_bbHeight; ++y) {
        int ey = (y * ENGINE_H) / g_bbHeight;
        int rowBase = ey * ENGINE_W;
        const uint8_t* srcRow = srcPtr + y * srcPitch;
        uint8_t*       dstRow = dstPtr + y * dstPitch;

        for (uint32_t x = 0; x < g_bbWidth; ++x) {
            float baseR, baseG, baseB;
            uint64_t rawOrig;
            read_dxgi_pixel(srcRow, x, g_dxgiFormat, baseR, baseG, baseB, rawOrig);

            int ex = (x * ENGINE_W) / g_bbWidth;
            int idx = rowBase + ex;

            float aiR = g_fp16ToFloat[outR[idx]];
            float aiG = g_fp16ToFloat[outG[idx]];
            float aiB = g_fp16ToFloat[outB[idx]];

            // Extract bounded HDR luminance boost from AI prediction so missing 3D normals never add colour noise
            float aiLuma = 0.2126f * std::fabs(aiR) + 0.7152f * std::fabs(aiG) + 0.0722f * std::fabs(aiB);
            float hdrBoost = aiLuma / (1.0f + aiLuma); // Smoothly bounded in [0.0, 1.0)

            // Apply AI-driven HDR highlight expansion & rich contrast curve to the true game colours
            float gain = 0.88f + 0.45f * hdrBoost;
            float finalR = std::pow(baseR, 0.92f) * gain;
            float finalG = std::pow(baseG, 0.92f) * gain;
            float finalB = std::pow(baseB, 0.92f) * gain;

            write_dxgi_pixel(dstRow, x, g_dxgiFormat, finalR, finalG, finalB, rawOrig);
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

    reshade::api::command_list* cmd = queue->get_immediate_command_list();
    reshade::api::resource backbuffer = swapchain->get_current_back_buffer();

    if (g_isD3D12 && g_d3d12ReadbackBuf && g_d3d12UploadBuf && g_mappedReadPtr && g_mappedWritePtr) {
        ID3D12GraphicsCommandList* d3d12Cmd = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native());
        ID3D12Resource* bbRes = reinterpret_cast<ID3D12Resource*>(backbuffer.handle);

        D3D12_RESOURCE_BARRIER barrier = {};
        barrier.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource   = bbRes;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
        d3d12Cmd->ResourceBarrier(1, &barrier);

        D3D12_TEXTURE_COPY_LOCATION srcLoc = {};
        srcLoc.pResource        = bbRes;
        srcLoc.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION dstLoc = {};
        dstLoc.pResource       = g_d3d12ReadbackBuf;
        dstLoc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dstLoc.PlacedFootprint = g_d3d12Footprint;

        d3d12Cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);

        queue->flush_immediate_command_list();
        queue->wait_idle();

        process_frame_pixels(g_mappedReadPtr, g_d3d12Footprint.Footprint.RowPitch, g_mappedWritePtr, g_d3d12Footprint.Footprint.RowPitch);

        cmd = queue->get_immediate_command_list();
        d3d12Cmd = reinterpret_cast<ID3D12GraphicsCommandList*>(cmd->get_native());

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
        d3d12Cmd->ResourceBarrier(1, &barrier);

        D3D12_TEXTURE_COPY_LOCATION uploadSrc = {};
        uploadSrc.pResource       = g_d3d12UploadBuf;
        uploadSrc.Type            = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        uploadSrc.PlacedFootprint = g_d3d12Footprint;

        D3D12_TEXTURE_COPY_LOCATION bbDst = {};
        bbDst.pResource        = bbRes;
        bbDst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        bbDst.SubresourceIndex = 0;

        d3d12Cmd->CopyTextureRegion(&bbDst, 0, 0, 0, &uploadSrc, nullptr);

        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter  = D3D12_RESOURCE_STATE_PRESENT;
        d3d12Cmd->ResourceBarrier(1, &barrier);

        queue->flush_immediate_command_list();
    }
}

static void on_destroy_swapchain(reshade::api::swapchain* swapchain, bool resize) {
    if (g_d3d12ReadbackBuf) { g_d3d12ReadbackBuf->Unmap(0, nullptr); g_d3d12ReadbackBuf->Release(); g_d3d12ReadbackBuf = nullptr; g_mappedReadPtr = nullptr; }
    if (g_d3d12UploadBuf)   { g_d3d12UploadBuf->Unmap(0, nullptr);   g_d3d12UploadBuf->Release();   g_d3d12UploadBuf   = nullptr; g_mappedWritePtr = nullptr; }
    if (fn_cudaFree) {
        if (g_dInput)  { fn_cudaFree(g_dInput);  g_dInput  = nullptr; }
        if (g_dOutput) { fn_cudaFree(g_dOutput); g_dOutput = nullptr; }
    }
}

static void on_destroy_device(reshade::api::device* device) {
    if (fn_cudaStreamSynchronize && g_cudaStream) fn_cudaStreamSynchronize(g_cudaStream);
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
