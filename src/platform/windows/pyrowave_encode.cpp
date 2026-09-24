/**
 * @file src/platform/windows/pyrowave_encode.cpp
 * @brief PyroWave host encoder on Direct3D 12. See pyrowave_encode.h for the frame path.
 */
#ifdef SUNSHINE_ENABLE_PYROWAVE

  // standard includes
  #include <algorithm>
  #include <chrono>
  #include <cstring>
  #include <map>
  #include <optional>
  #include <string>

  // platform includes
  #include <d3d11_4.h>
  #include <d3d12.h>
  #include <d3dcompiler.h>
  #include <dxgi1_4.h>
  #include <wrl/client.h>

  // lib includes
  #include <pyrowave_d3d12.h>

  // local includes
  #include "display_vram.h"
  #include "pyrowave_encode.h"
  #include "src/logging.h"
  #include "src/utility.h"
  #include "src/video.h"

  #if !defined(SUNSHINE_SHADERS_DIR)
    #define SUNSHINE_SHADERS_DIR SUNSHINE_ASSETS_DIR "/shaders/directx"
  #endif

using Microsoft::WRL::ComPtr;
using namespace std::literals;

namespace platf::pyrowave {

  namespace {
    // Packets are sized for the UDP payload; a packet may exceed this by up to one coded
    // block (PyroWave's packetizer only closes a packet after the block that overflows it).
    constexpr std::size_t packet_boundary = 1024;

    // The capture-rate budget (see encoder_t::on_new_capture) grows frames by at most this
    // factor over bitrate / stream fps, and never past max_boosted_frame_bytes. Beyond
    // about 1 MB (tail over 3 x 255 packets of ~1.2 KB) stream.cpp can no longer split the
    // frame into a protected head and bare tail, and its fallback drops FEC entirely.
    constexpr double max_budget_boost = 2.0;
    constexpr std::size_t max_boosted_frame_bytes = 900 * 1000;

    // Smoothing of the measured capture interval, per captured frame.
    constexpr double capture_interval_smoothing = 0.1;

    // Bands (see pyrowave_d3d12_encoder_get_num_active_blocks) covering the sequence of
    // wavelet levels 4 and 3: the loss-critical head the RTP layer protects with FEC.
    constexpr int fec_head_bands = 3;

    void log_message(void *, const char *msg) {
      BOOST_LOG(info) << "PyroWave: "sv << msg;
    }

    const char *result_string(pyrowave_d3d12_result result) {
      return pyrowave_d3d12_result_to_string(result);
    }

    // Transfer function variants of convert_pyrowave_cs.hlsl.
    enum class transfer_e {
      unorm,  ///< UNORM capture: values are already display encoded
      linear_sdr,  ///< FP16 scRGB capture of SDR content: apply the sRGB curve
      pq,  ///< FP16 scRGB capture of HDR content: Rec. 2100 PQ
    };

    ComPtr<ID3D11ComputeShader> compile_convert_shader(ID3D11Device *device, transfer_e transfer, bool subsample_420) {
      std::vector<D3D_SHADER_MACRO> macros;
      if (transfer == transfer_e::pq) {
        macros.push_back({"PQ", "1"});
      } else if (transfer == transfer_e::linear_sdr) {
        macros.push_back({"LINEAR", "1"});
      }
      if (subsample_420) {
        macros.push_back({"SUBSAMPLE_420", "1"});
      }
      macros.push_back({nullptr, nullptr});

      const std::string path = SUNSHINE_SHADERS_DIR "/convert_pyrowave_cs.hlsl";
      const std::wstring wpath(path.begin(), path.end());
      ComPtr<ID3DBlob> code, errors;
      HRESULT hr = D3DCompileFromFile(wpath.c_str(), macros.data(), D3D_COMPILE_STANDARD_FILE_INCLUDE, "main_cs", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
      if (FAILED(hr)) {
        BOOST_LOG(error) << "PyroWave: compiling "sv << path << " failed [0x"sv << util::hex(hr).to_string_view() << "]: "sv
                         << (errors ? std::string_view((const char *) errors->GetBufferPointer(), errors->GetBufferSize()) : "no details"sv);
        return {};
      }

      ComPtr<ID3D11ComputeShader> shader;
      hr = device->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &shader);
      if (FAILED(hr)) {
        BOOST_LOG(error) << "PyroWave: CreateComputeShader failed [0x"sv << util::hex(hr).to_string_view() << ']';
        return {};
      }
      return shader;
    }

    ComPtr<IDXGIAdapter> adapter_of(ID3D11Texture2D *texture) {
      ComPtr<ID3D11Device> device;
      texture->GetDevice(&device);
      ComPtr<IDXGIDevice> dxgi_device;
      ComPtr<IDXGIAdapter> adapter;
      if (device && SUCCEEDED(device.As(&dxgi_device))) {
        dxgi_device->GetAdapter(&adapter);
      }
      return adapter;
    }

    // Matches the params_cbuffer in convert_pyrowave_cs.hlsl.
    struct convert_params_t {
      uint32_t luma_size[2];
      float content_offset[2];
      float content_scale[2];
      uint32_t blank;
      uint32_t padding;
    };

    static_assert(sizeof(convert_params_t) % 16 == 0, "Constant buffers are sized in 16-byte units");
  }  // namespace

  bool validate() {
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
      return false;
    }

    for (UINT i = 0;; i++) {
      ComPtr<IDXGIAdapter1> adapter;
      if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) {
        break;
      }
      DXGI_ADAPTER_DESC1 desc {};
      adapter->GetDesc1(&desc);
      if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
        continue;
      }

      ComPtr<ID3D12Device> device;
      if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) &&
          pyrowave_d3d12_device_supports_encoder(device.Get())) {
        return true;
      }
    }
    return false;
  }

  struct encoder_t::impl_t {
    enum class state_e {
      pending,  ///< GPU resources are created on the first frame
      ready,
      failed,
    };

    // A captured image opened on the conversion device, cached by img_d3d_t::id.
    struct img_ctx_t {
      ComPtr<ID3D11Texture2D> texture;
      ComPtr<IDXGIKeyedMutex> mutex;
      ComPtr<ID3D11ShaderResourceView> srv;
      ID3D11Texture2D *capture_texture_p = nullptr;
      std::weak_ptr<const platf::img_t> img_weak;
    };

    state_e state = state_e::pending;

    int width = 0;
    int height = 0;
    bool yuv444 = false;
    ::video::sunshine_colorspace_t colorspace {};
    int framerate = 60;
    int bitrate_kbps = 0;
    std::size_t max_bitstream = 0;

    // Smoothed interval between newly captured frames, in seconds.
    double capture_interval = 0.0;
    std::optional<std::chrono::steady_clock::time_point> last_capture;

    void set_bitrate(int kbps) {
      bitrate_kbps = kbps;
      update_budget();
    }

    void on_new_capture(std::chrono::steady_clock::time_point when) {
      const double nominal = 1.0 / framerate;
      if (last_capture) {
        // Bounded, so a pause (static screen, loading) cannot drag the average out.
        const double dt = std::clamp(std::chrono::duration<double>(when - *last_capture).count(), nominal, nominal * max_budget_boost);
        capture_interval += capture_interval_smoothing * (dt - capture_interval);
      }
      last_capture = when;
      update_budget();
    }

    // Intra only: every frame gets bitrate / (frames encoded per second).
    void update_budget() {
      const std::size_t nominal = (std::size_t) ((int64_t) bitrate_kbps * 1000 / framerate / 8);
      std::size_t budget = (std::size_t) ((double) bitrate_kbps * 1000.0 * capture_interval / 8.0);
      budget = std::min(budget, std::max(nominal, max_boosted_frame_bytes));
      max_bitstream = std::max<std::size_t>(4096, std::max(budget, nominal));
    }

    // Conversion, on a private D3D11 device on the capture adapter.
    ComPtr<ID3D11Device5> device11;
    ComPtr<ID3D11DeviceContext4> context11;
    ComPtr<ID3D11Texture2D> planes11[3];
    ComPtr<ID3D11UnorderedAccessView> plane_uavs[3];
    ComPtr<ID3D11Buffer> color_matrix;
    ComPtr<ID3D11Buffer> params;
    ComPtr<ID3D11SamplerState> sampler;
    std::map<transfer_e, ComPtr<ID3D11ComputeShader>> shaders;
    std::map<uint32_t, img_ctx_t> img_ctxs;
    convert_params_t last_params {};
    bool warned_unorm_hdr = false;

    // D3D11 -> D3D12 handoff: the planes, and a fence signalled once they are written.
    ComPtr<ID3D11Fence> convert_fence;
    ComPtr<ID3D12Fence> convert_fence12;
    uint64_t convert_value = 0;

    // Encoding.
    ComPtr<ID3D12Device> device12;
    ComPtr<ID3D12Resource> planes12[3];
    pyrowave_d3d12_device pw_device = nullptr;
    pyrowave_d3d12_encoder pw_encoder = nullptr;
    uint32_t head_block_end = 0;

    std::vector<pyrowave_d3d12_packet> packets;
    std::vector<uint8_t> scratch;

    ~impl_t() {
      if (pw_encoder) {
        pyrowave_d3d12_encoder_destroy(pw_encoder);
      }
      if (pw_device) {
        pyrowave_d3d12_device_destroy(pw_device);
      }
    }

    bool init_gpu(dxgi::img_d3d_t &img) {
      auto adapter = adapter_of(img.capture_texture.get());
      if (!adapter) {
        BOOST_LOG(error) << "PyroWave: could not determine the capture adapter"sv;
        return false;
      }

      // Conversion device.
      const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
      ComPtr<ID3D11Device> device;
      ComPtr<ID3D11DeviceContext> context;
      HRESULT hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, levels, (UINT) std::size(levels), D3D11_SDK_VERSION, &device, nullptr, &context);
      if (FAILED(hr) || FAILED(device.As(&device11)) || FAILED(context.As(&context11))) {
        BOOST_LOG(error) << "PyroWave: D3D11 conversion device (ID3D11Device5) unavailable [0x"sv << util::hex(hr).to_string_view() << ']';
        return false;
      }

      // Encoder.
      hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device12));
      if (FAILED(hr)) {
        BOOST_LOG(error) << "PyroWave: D3D12CreateDevice failed [0x"sv << util::hex(hr).to_string_view() << ']';
        return false;
      }

      pyrowave_d3d12_device_create_info device_info {};
      device_info.d3d12_device = device12.Get();
      device_info.message_callback = log_message;
      auto res = pyrowave_d3d12_device_create(&device_info, &pw_device);
      if (res != PYROWAVE_D3D12_SUCCESS) {
        BOOST_LOG(error) << "PyroWave: device creation failed: "sv << result_string(res);
        return false;
      }

      pyrowave_d3d12_encoder_create_info encoder_info {};
      encoder_info.device = pw_device;
      encoder_info.width = width;
      encoder_info.height = height;
      encoder_info.chroma = yuv444 ? PYROWAVE_D3D12_CHROMA_SUBSAMPLING_444 : PYROWAVE_D3D12_CHROMA_SUBSAMPLING_420;
      res = pyrowave_d3d12_encoder_create(&encoder_info, &pw_encoder);
      if (res != PYROWAVE_D3D12_SUCCESS) {
        BOOST_LOG(error) << "PyroWave: encoder creation failed: "sv << result_string(res);
        return false;
      }

      std::size_t head_blocks = 0;
      pyrowave_d3d12_encoder_get_num_active_blocks(pw_encoder, fec_head_bands, &head_blocks);
      head_block_end = (uint32_t) head_blocks;

      // The three planes: written here, read by the D3D12 encoder.
      const DXGI_FORMAT plane_format = colorspace.bit_depth > 8 ? DXGI_FORMAT_R16_UNORM : DXGI_FORMAT_R8_UNORM;
      for (int i = 0; i < 3; i++) {
        D3D11_TEXTURE2D_DESC desc {};
        desc.Width = (UINT) ((i == 0 || yuv444) ? width : width / 2);
        desc.Height = (UINT) ((i == 0 || yuv444) ? height : height / 2);
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = plane_format;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE;
        desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED;
        hr = device11->CreateTexture2D(&desc, nullptr, &planes11[i]);
        if (FAILED(hr)) {
          BOOST_LOG(error) << "PyroWave: plane texture creation failed [0x"sv << util::hex(hr).to_string_view() << ']';
          return false;
        }
        hr = device11->CreateUnorderedAccessView(planes11[i].Get(), nullptr, &plane_uavs[i]);
        if (FAILED(hr)) {
          BOOST_LOG(error) << "PyroWave: plane UAV creation failed [0x"sv << util::hex(hr).to_string_view() << ']';
          return false;
        }

        ComPtr<IDXGIResource1> resource;
        HANDLE handle = nullptr;
        hr = planes11[i].As(&resource);
        if (SUCCEEDED(hr)) {
          hr = resource->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &handle);
        }
        if (SUCCEEDED(hr)) {
          hr = device12->OpenSharedHandle(handle, IID_PPV_ARGS(&planes12[i]));
        }
        if (handle) {
          CloseHandle(handle);
        }
        if (FAILED(hr)) {
          BOOST_LOG(error) << "PyroWave: sharing plane "sv << i << " with D3D12 failed [0x"sv << util::hex(hr).to_string_view() << ']';
          return false;
        }
      }

      // Fence the D3D12 encoder waits on for the conversion.
      HANDLE fence_handle = nullptr;
      hr = device11->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&convert_fence));
      if (SUCCEEDED(hr)) {
        hr = convert_fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &fence_handle);
      }
      if (SUCCEEDED(hr)) {
        hr = device12->OpenSharedHandle(fence_handle, IID_PPV_ARGS(&convert_fence12));
      }
      if (fence_handle) {
        CloseHandle(fence_handle);
      }
      if (FAILED(hr)) {
        BOOST_LOG(error) << "PyroWave: sharing the conversion fence failed [0x"sv << util::hex(hr).to_string_view() << ']';
        return false;
      }

      // Colour matrix: the same vectors the other encoders' converters use.
      const auto *color_vectors = ::video::color_vectors_from_colorspace(colorspace, true);
      if (!color_vectors) {
        BOOST_LOG(error) << "PyroWave: no colour vectors for the stream colorspace"sv;
        return false;
      }
      D3D11_BUFFER_DESC cb {};
      cb.ByteWidth = (UINT) ((sizeof(*color_vectors) + 15) & ~15u);
      cb.Usage = D3D11_USAGE_IMMUTABLE;
      cb.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
      D3D11_SUBRESOURCE_DATA init {color_vectors, 0, 0};
      hr = device11->CreateBuffer(&cb, &init, &color_matrix);
      if (FAILED(hr)) {
        return false;
      }
      cb.ByteWidth = sizeof(convert_params_t);
      cb.Usage = D3D11_USAGE_DEFAULT;
      hr = device11->CreateBuffer(&cb, nullptr, &params);
      if (FAILED(hr)) {
        return false;
      }

      D3D11_SAMPLER_DESC sd {};
      sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
      sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
      sd.MaxLOD = D3D11_FLOAT32_MAX;
      hr = device11->CreateSamplerState(&sd, &sampler);
      return SUCCEEDED(hr);
    }

    ID3D11ComputeShader *shader_for(transfer_e transfer) {
      auto &shader = shaders[transfer];
      if (!shader) {
        shader = compile_convert_shader(device11.Get(), transfer, !yuv444);
      }
      return shader.Get();
    }

    img_ctx_t *open_image(dxgi::img_d3d_t &img) {
      // Forget images the capture side has released.
      for (auto it = img_ctxs.begin(); it != img_ctxs.end();) {
        it = it->second.img_weak.expired() ? img_ctxs.erase(it) : std::next(it);
      }

      auto &ctx = img_ctxs[img.id];
      if (ctx.texture && ctx.capture_texture_p == img.capture_texture.get()) {
        return &ctx;
      }

      ctx = {};
      HRESULT hr = device11->OpenSharedResource1(img.encoder_texture_handle, IID_PPV_ARGS(&ctx.texture));
      if (SUCCEEDED(hr)) {
        hr = ctx.texture.As(&ctx.mutex);
      }
      if (SUCCEEDED(hr)) {
        hr = device11->CreateShaderResourceView(ctx.texture.Get(), nullptr, &ctx.srv);
      }
      if (FAILED(hr)) {
        BOOST_LOG(error) << "PyroWave: opening the captured frame failed [0x"sv << util::hex(hr).to_string_view() << ']';
        img_ctxs.erase(img.id);
        return nullptr;
      }
      ctx.capture_texture_p = img.capture_texture.get();
      ctx.img_weak = img.weak_from_this();
      return &ctx;
    }

    void update_params(const D3D11_TEXTURE2D_DESC *input, bool blank) {
      convert_params_t p {};
      p.luma_size[0] = (uint32_t) width;
      p.luma_size[1] = (uint32_t) height;
      p.blank = blank ? 1 : 0;
      if (input && input->Width && input->Height) {
        // Letterbox the picture into the stream extent, preserving its aspect ratio.
        const float scale = std::min((float) width / (float) input->Width, (float) height / (float) input->Height);
        const float content_w = (float) input->Width * scale;
        const float content_h = (float) input->Height * scale;
        p.content_offset[0] = ((float) width - content_w) * 0.5f;
        p.content_offset[1] = ((float) height - content_h) * 0.5f;
        p.content_scale[0] = 1.0f / content_w;
        p.content_scale[1] = 1.0f / content_h;
      }
      if (std::memcmp(&p, &last_params, sizeof(p)) != 0) {
        context11->UpdateSubresource(params.Get(), 0, nullptr, &p, 0, 0);
        last_params = p;
      }
    }

    /**
     * @brief Convert the captured frame into the planes and signal the handoff fence.
     * @return 0 on success, positive to skip the frame, negative on a fatal error.
     */
    int convert(dxgi::img_d3d_t &img) {
      img_ctx_t *ctx = nullptr;
      transfer_e transfer = transfer_e::unorm;
      const bool blank = img.blank || !img.capture_texture;

      if (!blank) {
        ctx = open_image(img);
        if (!ctx) {
          return -1;
        }

        if (img.format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
          transfer = ::video::colorspace_is_hdr(colorspace) ? transfer_e::pq : transfer_e::linear_sdr;
        } else if (::video::colorspace_is_hdr(colorspace) && !warned_unorm_hdr) {
          BOOST_LOG(warning) << "PyroWave: HDR stream with a non-FP16 capture; encoding the capture values as-is"sv;
          warned_unorm_hdr = true;
        }

        D3D11_TEXTURE2D_DESC input_desc;
        ctx->texture->GetDesc(&input_desc);
        update_params(&input_desc, false);
      } else {
        update_params(nullptr, true);
      }

      auto *shader = shader_for(transfer);
      if (!shader) {
        return -1;
      }

      // Synchronize with the capture side, with a finite timeout so display re-init or
      // device loss cannot wedge the stream.
      if (ctx) {
        const HRESULT status = ctx->mutex->AcquireSync(0, 3000);
        if (status == WAIT_TIMEOUT) {
          BOOST_LOG(warning) << "PyroWave: timed out acquiring the capture mutex; skipping frame"sv;
          return 1;
        }
        if (status != S_OK && status != WAIT_ABANDONED) {
          BOOST_LOG(error) << "PyroWave: acquiring the capture mutex failed [0x"sv << util::hex(status).to_string_view() << ']';
          return 1;
        }
      }

      ID3D11Buffer *cbs[] = {color_matrix.Get(), params.Get()};
      ID3D11ShaderResourceView *srv = ctx ? ctx->srv.Get() : nullptr;
      ID3D11UnorderedAccessView *uavs[] = {plane_uavs[0].Get(), plane_uavs[1].Get(), plane_uavs[2].Get()};
      ID3D11SamplerState *samplers[] = {sampler.Get()};
      context11->CSSetShader(shader, nullptr, 0);
      context11->CSSetConstantBuffers(0, 2, cbs);
      context11->CSSetShaderResources(0, 1, &srv);
      context11->CSSetSamplers(0, 1, samplers);
      context11->CSSetUnorderedAccessViews(0, 3, uavs, nullptr);

      const UINT work_w = (UINT) (yuv444 ? width : (width + 1) / 2);
      const UINT work_h = (UINT) (yuv444 ? height : (height + 1) / 2);
      context11->Dispatch((work_w + 7) / 8, (work_h + 7) / 8, 1);

      ID3D11ShaderResourceView *null_srv = nullptr;
      ID3D11UnorderedAccessView *null_uavs[3] = {};
      context11->CSSetShaderResources(0, 1, &null_srv);
      context11->CSSetUnorderedAccessViews(0, 3, null_uavs, nullptr);

      // The copy of the frame is in the planes as far as D3D11 ordering goes, so the
      // capture side can have its texture back.
      if (ctx) {
        ctx->mutex->ReleaseSync(0);
      }

      context11->Signal(convert_fence.Get(), ++convert_value);
      context11->Flush();
      return 0;
    }
  };

  std::unique_ptr<encoder_t> encoder_t::create(const ::video::config_t &config, const ::video::sunshine_colorspace_t &colorspace) {
    const bool yuv444 = config.chromaSamplingType == 1;
    int width = config.width;
    int height = config.height;
    if (!yuv444) {
      // 4:2:0 needs even dimensions.
      width &= ~1;
      height &= ~1;
    }
    if (width <= 0 || height <= 0) {
      return nullptr;
    }

    auto self = std::unique_ptr<encoder_t>(new encoder_t());
    self->impl = std::make_unique<impl_t>();
    auto &impl = *self->impl;
    impl.width = width;
    impl.height = height;
    impl.yuv444 = yuv444;
    impl.colorspace = colorspace;

    impl.framerate = config.framerate > 0 ? config.framerate : 60;
    impl.capture_interval = 1.0 / impl.framerate;
    impl.set_bitrate(config.bitrate);

    BOOST_LOG(info) << "PyroWave encoder session: "sv << width << 'x' << height << (yuv444 ? " 4:4:4"sv : " 4:2:0"sv)
                    << ", "sv << colorspace.bit_depth << "-bit"sv << (::video::colorspace_is_hdr(colorspace) ? " HDR"sv : " SDR"sv)
                    << ", budget "sv << impl.max_bitstream << " bytes/frame"sv;
    return self;
  }

  int encoder_t::encode(platf::img_t &img_base, std::vector<uint8_t> &out, std::size_t &fec_head_bytes) {
    auto &impl = *this->impl;
    fec_head_bytes = 0;

    if (impl.state == impl_t::state_e::failed) {
      return -1;
    }

    auto *img = dynamic_cast<dxgi::img_d3d_t *>(&img_base);
    if (!img) {
      BOOST_LOG(error) << "PyroWave requires a GPU (VRAM) capture"sv;
      impl.state = impl_t::state_e::failed;
      return -1;
    }

    if (impl.state == impl_t::state_e::pending) {
      if (!img->capture_texture) {
        // Wait for a real capture: it tells us the adapter.
        return 1;
      }
      if (!impl.init_gpu(*img)) {
        impl.state = impl_t::state_e::failed;
        return -1;
      }
      impl.state = impl_t::state_e::ready;
      BOOST_LOG(info) << "PyroWave encoder ready (Direct3D 12)"sv;
    }

    const int converted = impl.convert(*img);
    if (converted != 0) {
      if (converted < 0) {
        impl.state = impl_t::state_e::failed;
      }
      return converted;
    }

    // Encode on the GPU once the conversion has landed.
    pyrowave_d3d12_encoder_gpu_input input {};
    for (int i = 0; i < 3; i++) {
      input.planes[i] = impl.planes12[i].Get();
    }
    input.wait_fence = impl.convert_fence12.Get();
    input.wait_value = impl.convert_value;

    pyrowave_d3d12_rate_control rc {};
    rc.maximum_bitstream_size = impl.max_bitstream;

    auto res = pyrowave_d3d12_encoder_encode_gpu(impl.pw_encoder, &input, &rc);
    std::size_t num_packets = 0;
    if (res == PYROWAVE_D3D12_SUCCESS) {
      res = pyrowave_d3d12_encoder_compute_num_packets(impl.pw_encoder, packet_boundary, &num_packets);
    }
    if (res != PYROWAVE_D3D12_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: encode failed: "sv << result_string(res);
      impl.state = impl_t::state_e::failed;
      return -1;
    }

    // The packetizer does not bound-check its output, and a packet can overshoot the
    // boundary by one coded block (payload_words is 12 bits: at most ~16 KiB).
    constexpr std::size_t max_block_bytes = 64 * 1024;
    const std::size_t scratch_bound = 4096 + num_packets * (packet_boundary + max_block_bytes);
    if (impl.scratch.size() < scratch_bound) {
      impl.scratch.resize(scratch_bound);
    }
    if (impl.packets.size() < num_packets) {
      impl.packets.resize(num_packets);
    }
    std::size_t out_packets = 0;
    res = pyrowave_d3d12_encoder_packetize(impl.pw_encoder, impl.packets.data(), packet_boundary, &out_packets, impl.scratch.data(), impl.scratch.size());
    if (res != PYROWAVE_D3D12_SUCCESS) {
      BOOST_LOG(error) << "PyroWave: packetize failed: "sv << result_string(res);
      return 1;
    }

    // PyroWave frames are several independently decodable packets, but the RTP layer
    // ships one opaque payload per frame, so frame them for the client:
    //   [u32 packet_count] { [u32 size] [size bytes] } * packet_count
    auto put_u32 = [&out](uint32_t v) {
      out.push_back((uint8_t) (v & 0xff));
      out.push_back((uint8_t) ((v >> 8) & 0xff));
      out.push_back((uint8_t) ((v >> 16) & 0xff));
      out.push_back((uint8_t) ((v >> 24) & 0xff));
    };

    // Block index of a packet's first block: every block starts with an 8-byte
    // BitstreamHeader whose second u32 is { quant_code : 8, block_index : 24 }.
    auto first_block_index = [&impl](std::size_t i) {
      const uint8_t *src = impl.scratch.data() + impl.packets[i].offset;
      const uint32_t w = (uint32_t) src[4] | ((uint32_t) src[5] << 8) | ((uint32_t) src[6] << 16) | ((uint32_t) src[7] << 24);
      return w >> 8;
    };

    // Sized up front: the caller hands over a fresh vector every frame, and growing it
    // packet by packet cost 0.1-0.3 ms per 460 KB frame (vs 0.01 ms) on this path.
    std::size_t framed_size = 4;
    for (std::size_t i = 0; i < out_packets; i++) {
      framed_size += 4 + impl.packets[i].size;
    }
    out.clear();
    out.reserve(framed_size);
    put_u32((uint32_t) out_packets);
    for (std::size_t i = 0; i < out_packets; i++) {
      // Blocks are emitted coarsest level first, so the protected head closes at the
      // first packet that starts past the last protected level. A packet straddling the
      // boundary stays whole in the head. Packet 0 leads with the sequence header rather
      // than a block header and always belongs to the head.
      if (fec_head_bytes == 0 && i > 0 && first_block_index(i) >= impl.head_block_end) {
        fec_head_bytes = out.size();
      }
      put_u32((uint32_t) impl.packets[i].size);
      const uint8_t *src = impl.scratch.data() + impl.packets[i].offset;
      out.insert(out.end(), src, src + impl.packets[i].size);
    }
    return 0;
  }

  void encoder_t::set_bitrate(int bitrate_kbps) {
    impl->set_bitrate(bitrate_kbps);
    BOOST_LOG(info) << "PyroWave: budget now "sv << impl->max_bitstream << " bytes/frame"sv;
  }

  void encoder_t::on_new_capture(std::chrono::steady_clock::time_point when) {
    const std::size_t before = impl->max_bitstream;
    impl->on_new_capture(when);
    if (impl->max_bitstream != before) {
      BOOST_LOG(debug) << "PyroWave: capture rate "sv << 1.0 / impl->capture_interval << " fps, budget "sv << impl->max_bitstream << " bytes/frame"sv;
    }
  }

  encoder_t::~encoder_t() = default;

}  // namespace platf::pyrowave

#endif  // SUNSHINE_ENABLE_PYROWAVE
