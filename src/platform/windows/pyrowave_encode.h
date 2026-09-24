/**
 * @file src/platform/windows/pyrowave_encode.h
 * @brief PyroWave (intra-only GPU wavelet codec) host encoder on Direct3D 12.
 *
 * PyroWave does not ride the avcodec/nvenc/AMF encoder_t abstraction: it produces an
 * already packetized bitstream that is shipped as packet_raw_generic. Frame path:
 *
 *   captured frame (D3D11 shared texture, keyed mutex)
 *     -> opened on a private D3D11 device, converted RGB -> Y'CbCr by a compute shader
 *        straight into three shared planes (no intermediate copy)
 *     -> D3D11 signals a shared fence
 *     -> the D3D12 PyroWave encoder (third-party/pyrowave, d3d12/) waits for it on the GPU,
 *        encodes and packetizes
 *
 * Every frame is intra (IDR). The bitstream is ordered most important first (coarse
 * wavelet levels lead), which the RTP layer uses to protect only the loss-critical head
 * with FEC; see video::packet_raw_t::fec_head_bytes.
 */
#pragma once

#ifdef SUNSHINE_ENABLE_PYROWAVE

  #include <chrono>
  #include <cstddef>
  #include <cstdint>
  #include <memory>
  #include <vector>

  #include "src/video_colorspace.h"

namespace platf {
  struct img_t;
}

namespace video {
  struct config_t;
}

namespace platf::pyrowave {

  /**
   * @brief Whether a hardware adapter can run the D3D12 PyroWave encoder.
   *
   * Needs Direct3D 12 with Shader Model 6.6, native 16-bit shader types and 64-wide waves
   * (any recent AMD/NVIDIA/Intel desktop GPU). Gates whether PyroWave is advertised.
   */
  bool validate();

  /**
   * @brief One PyroWave encode session.
   *
   * GPU resources are created on the first frame, once the capture adapter is known from
   * the captured image. Not thread-safe.
   */
  class encoder_t {
  public:
    ~encoder_t();

    /**
     * @param config Negotiated stream configuration (size, chroma subsampling, bitrate
     *               and frame rate, which set the per-frame byte budget).
     * @param colorspace Stream colorspace (colour matrix, range, bit depth, HDR).
     */
    static std::unique_ptr<encoder_t> create(const ::video::config_t &config, const ::video::sunshine_colorspace_t &colorspace);

    /**
     * @brief Encode one captured frame.
     *
     * @param img Captured image; must be a D3D11 (VRAM) capture.
     * @param out Receives the access unit: [u32 packet_count] { [u32 size] [bytes] } *,
     *            the framing moonlight PyroWave clients parse.
     * @param fec_head_bytes Receives the byte offset in @p out where the loss-critical
     *                       head (sequence header and wavelet levels 4 and 3) ends, or 0
     *                       when the frame is too small to split.
     * @return 0 on success; negative on a fatal error (end the session); positive when
     *         this frame was skipped but the session can continue.
     */
    int encode(platf::img_t &img, std::vector<uint8_t> &out, std::size_t &fec_head_bytes);

    /**
     * @brief Change the video bitrate (dynamic bitrate).
     * @param bitrate_kbps New video bitrate in kbps.
     */
    void set_bitrate(int bitrate_kbps);

    /**
     * @brief Report that a newly captured frame arrived (not a re-encode of the last one).
     *
     * Every frame is intra, so the per-frame byte budget is the bitrate divided by the
     * rate frames are actually encoded at. When the game renders below the stream frame
     * rate (60 fps in a 120 fps stream), dividing by the stream rate would leave part of
     * the bitrate unused; this measures the capture rate so it does not.
     *
     * @param when When the frame was received.
     */
    void on_new_capture(std::chrono::steady_clock::time_point when);

  private:
    encoder_t() = default;

    struct impl_t;
    std::unique_ptr<impl_t> impl;
  };

}  // namespace platf::pyrowave

#endif  // SUNSHINE_ENABLE_PYROWAVE
