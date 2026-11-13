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
#include <cmath>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

// ============================================================================
// 1. DYNAMIC CUDA & TENSORRT FUNCTION POINTERS
// ============================================================================
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

// Fast IEEE 754 Float32 <-> Float16 conversion
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x03ff;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | (exp << 10) | mant);
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
        exp++; mant &= ~0x0400;
    } else if (exp == 31) {
        return 0.0f; // Filter out NaN/Inf from FP16 overflow
    }
    exp = exp - 15 + 127;
    uint32_t x = sign | (exp << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

// ============================================================================
// 2. GLOBAL ENGINE & DIRECTX 12 STATE
// ============================================================================
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
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

static constexpr int ENGINE_W = 1920;
static constexpr int ENGINE_H = 1080;
static constexpr int ENGINE_PIXELS = ENGINE_W * ENGINE_H;

static void*                 g_dInput   = nullptr; // g_buffer (1x7x1080x1920 FP16)
static void*                 g_dAuxIn   = nullptr; // Extra buffer in case model has a 2nd input tensor
static void*                 g_dOutput  = nullptr; // rendered_frame (1x3x1080x1920 FP16)
static std::vector<uint16_t> g_cpuGbuf;
static std::vector<uint16_t> g_cpuOut;

static ID3D12Resource*                    g_d3d12ReadbackBuf = nullptr;
static ID3D12Resource*                    g_d3d12UploadBuf   = nullptr;
static D3D12_PLACED_SUBRESOURCE_FOOTPRINT g_d3d12Footprint   = {};
static uint8_t*                           g_mappedReadPtr    = nullptr;
static uint8_t*                           g_mappedWritePtr   = nullptr;

static reshade::api::resource g_texRead  = { 0 };
static reshade::api::resource g_texWrite = { 0 };
static bool                   g_isD3D12  = false;
static uint32_t               g_bbWidth  = 1920;
static uint32_t               g_bbHeight = 1080;
static reshade::api::format   g_bbFormat = reshade::api::format::r8g8b8a8_unorm;

static bool load_libraries() {
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
    reshade::api::resource_desc desc  = dev->get_resource_desc(backbuffer);
    g_bbWidth  = desc.texture.width  ? desc.texture.width  : 1920;
    g_bbHeight = desc.texture.height ? desc.texture.height : 1080;
    g_bbFormat = desc.texture.format;

    fn_cudaMalloc(&g_dInput,  (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t));
    fn_cudaMalloc(&g_dAuxIn,  (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t));
    fn_cudaMalloc(&g_dOutput, (size_t)ENGINE_PIXELS * 3 * sizeof(uint16_t));
    if (fn_cudaMemset) {
        fn_cudaMemset(g_dInput,  0, (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t));
        fn_cudaMemset(g_dAuxIn,  0, (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t));
        fn_cudaMemset(g_dOutput, 0, (size_t)ENGINE_PIXELS * 3 * sizeof(uint16_t));
    }

    g_cpuGbuf.assign((size_t)ENGINE_PIXELS * 7, 0);
    g_cpuOut.assign((size_t)ENGINE_PIXELS * 3, 0);

    // Fix Bug 1: Bind all inputs and outputs by exact name/role (handles 2-input + 1-output models!)
    int32_t nbTensors = g_trtEngine->getNbIOTensors();
    bool boundPrimaryInput = false;
    for (int32_t i = 0; i < nbTensors; ++i) {
        const char* name = g_trtEngine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = g_trtEngine->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims dims = g_trtEngine->getTensorShape(name);
            if (dims.nbDims == 4) {
                nvinfer1::Dims runDims;
                runDims.nbDims = 4;
                runDims.d[0] = (dims.d[0] > 0) ? dims.d[0] : 1;
                runDims.d[1] = (dims.d[1] > 0) ? dims.d[1] : 7;
                runDims.d[2] = ENGINE_H;
                runDims.d[3] = ENGINE_W;
                g_trtContext->setInputShape(name, runDims);
            }
            if (std::strcmp(name, "g_buffer") == 0 || !boundPrimaryInput) {
                g_trtContext->setTensorAddress(name, g_dInput);
                boundPrimaryInput = true;
            } else {
                g_trtContext->setTensorAddress(name, g_dAuxIn);
            }
        } else {
            g_trtContext->setTensorAddress(name, g_dOutput);
        }
    }

    if (g_isD3D12) {
        ID3D12Device* d3d12Dev = reinterpret_cast<ID3D12Device*>(dev->get_native());
        ID3D12Resource* bbRes  = reinterpret_cast<ID3D12Resource*>(backbuffer.handle);
        D3D12_RESOURCE_DESC bbDesc = bbRes->GetDesc();

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
    } else {
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
    bool isBGRA = (g_bbFormat == reshade::api::format::b8g8r8a8_unorm ||
                   g_bbFormat == reshade::api::format::b8g8r8a8_unorm_srgb ||
                   g_bbFormat == reshade::api::format::b8g8r8x8_unorm ||
                   g_bbFormat == reshade::api::format::b8g8r8x8_unorm_srgb);
    int rIdx = isBGRA ? 2 : 0;
    int gIdx = 1;
    int bIdx = isBGRA ? 0 : 2;

    uint32_t copyW = (g_bbWidth  < ENGINE_W) ? g_bbWidth  : ENGINE_W;
    uint32_t copyH = (g_bbHeight < ENGINE_H) ? g_bbHeight : ENGINE_H;

    // 1. Pack actual window pixels into 1x7x1080x1920 FP16 G-Buffer
    // Channels 0,1,2: RGB [0..1]
    // Channels 3,4,5: Approximate Normals (0.5, 0.5, 1.0 flat camera-facing normal)
    // Channel  6:     Luminance / Depth [0..1]
    uint16_t flatNx = f32_to_f16(0.0f);
    uint16_t flatNy = f32_to_f16(0.0f);
    uint16_t flatNz = f32_to_f16(1.0f);

    for (uint32_t y = 0; y < copyH; ++y) {
        const uint8_t* row = srcPtr + y * srcPitch;
        for (uint32_t x = 0; x < copyW; ++x) {
            int idx = y * ENGINE_W + x;
            float r = row[x * 4 + rIdx] * (1.0f / 255.0f);
            float g = row[x * 4 + gIdx] * (1.0f / 255.0f);
            float b = row[x * 4 + bIdx] * (1.0f / 255.0f);
            float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;

            g_cpuGbuf[0 * ENGINE_PIXELS + idx] = f32_to_f16(r);
            g_cpuGbuf[1 * ENGINE_PIXELS + idx] = f32_to_f16(g);
            g_cpuGbuf[2 * ENGINE_PIXELS + idx] = f32_to_f16(b);
            g_cpuGbuf[3 * ENGINE_PIXELS + idx] = flatNx;
            g_cpuGbuf[4 * ENGINE_PIXELS + idx] = flatNy;
            g_cpuGbuf[5 * ENGINE_PIXELS + idx] = flatNz;
            g_cpuGbuf[6 * ENGINE_PIXELS + idx] = f32_to_f16(luma);
        }
    }

    // 2. Run TensorRT Inference
    fn_cudaMemcpyAsync(g_dInput, g_cpuGbuf.data(), (size_t)ENGINE_PIXELS * 7 * sizeof(uint16_t), cudaMemcpyHostToDevice, g_cudaStream);
    bool ok = g_trtContext->enqueueV3(g_cudaStream);
    fn_cudaMemcpyAsync(g_cpuOut.data(), g_dOutput, (size_t)ENGINE_PIXELS * 3 * sizeof(uint16_t), cudaMemcpyDeviceToHost, g_cudaStream);
    fn_cudaStreamSynchronize(g_cudaStream);

    if (!ok) {
        // If inference failed, preserve original frame so screen never turns blue
        for (uint32_t y = 0; y < g_bbHeight; ++y) {
            std::memcpy(dstPtr + y * dstPitch, srcPtr + y * srcPitch, g_bbWidth * 4);
        }
        return;
    }

    // 3. Unpack rendered_frame [1, 3, 1080, 1920] FP16 back to Game Backbuffer Format
    for (uint32_t y = 0; y < copyH; ++y) {
        const uint8_t* srcRow = srcPtr + y * srcPitch;
        uint8_t*       dstRow = dstPtr + y * dstPitch;
        for (uint32_t x = 0; x < copyW; ++x) {
            int idx = y * ENGINE_W + x;

            float nr = f16_to_f32(g_cpuOut[0 * ENGINE_PIXELS + idx]);
            float ng = f16_to_f32(g_cpuOut[1 * ENGINE_PIXELS + idx]);
            float nb = f16_to_f32(g_cpuOut[2 * ENGINE_PIXELS + idx]);

            // Reinhard HDR tone-map for values > 1.0 and clamp negative values
            nr = nr < 0.0f ? 0.0f : (nr / (1.0f + nr) * 1.4f);
            ng = ng < 0.0f ? 0.0f : (ng / (1.0f + ng) * 1.4f);
            nb = nb < 0.0f ? 0.0f : (nb / (1.0f + nb) * 1.4f);

            nr = nr > 1.0f ? 1.0f : nr;
            ng = ng > 1.0f ? 1.0f : ng;
            nb = nb > 1.0f ? 1.0f : nb;

            dstRow[x * 4 + rIdx] = (uint8_t)(nr * 255.0f + 0.5f);
            dstRow[x * 4 + gIdx] = (uint8_t)(ng * 255.0f + 0.5f);
            dstRow[x * 4 + bIdx] = (uint8_t)(nb * 255.0f + 0.5f);
            dstRow[x * 4 + 3]    = srcRow[x * 4 + 3];
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
        if (g_dInput)  { fn_cudaFree(g_dInput);  g_dInput  = nullptr; }
        if (g_dAuxIn)  { fn_cudaFree(g_dAuxIn);  g_dAuxIn  = nullptr; }
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
