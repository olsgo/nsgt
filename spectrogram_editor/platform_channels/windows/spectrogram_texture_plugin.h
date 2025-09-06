// Windows Platform Channel for GPU Texture Integration
// Provides native texture management and GPU interop for Flutter desktop

#pragma once

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>
#include <flutter/standard_method_codec.h>
#include <flutter/texture_registrar.h>

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <memory>
#include <map>

using Microsoft::WRL::ComPtr;

class SpectrogramTexturePlugin {
public:
    static void RegisterWithRegistrar(flutter::PluginRegistrarWindows* registrar);

    SpectrogramTexturePlugin(flutter::PluginRegistrarWindows* registrar);

    virtual ~SpectrogramTexturePlugin();

private:
    // Flutter method channel handling
    void HandleMethodCall(
        const flutter::MethodCall<flutter::EncodableValue>& method_call,
        std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);

    // Texture management
    int64_t CreateTexture(int width, int height);
    bool UpdateTexture(int64_t texture_id, const uint8_t* data, size_t data_size);
    bool DestroyTexture(int64_t texture_id);
    
    // GPU interop
    bool InitializeD3D();
    bool CreateSharedTexture(int width, int height, ComPtr<ID3D11Texture2D>& texture);
    bool UpdateSharedTexture(ComPtr<ID3D11Texture2D>& texture, 
                           const uint8_t* data, size_t data_size);

    // Flutter integration
    flutter::PluginRegistrarWindows* registrar_;
    std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>> channel_;
    flutter::TextureRegistrar* texture_registrar_;

    // DirectX resources
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    ComPtr<IDXGIFactory> dxgi_factory_;

    // Texture management
    std::map<int64_t, std::unique_ptr<flutter::TextureVariant>> textures_;
    int64_t next_texture_id_ = 1;
};

// Pixel buffer texture implementation for shared GPU memory
class PixelBufferTexture : public flutter::TextureVariant {
public:
    PixelBufferTexture(int width, int height);
    virtual ~PixelBufferTexture();

    // TextureVariant interface
    const FlutterDesktopPixelBuffer* CopyPixelBuffer(size_t width, size_t height) override;

    // Update methods
    bool UpdateFromSharedTexture(ComPtr<ID3D11Texture2D> shared_texture);
    bool UpdateFromRawData(const uint8_t* data, size_t data_size);

private:
    int width_;
    int height_;
    std::unique_ptr<uint8_t[]> pixel_buffer_;
    FlutterDesktopPixelBuffer flutter_pixel_buffer_;
    
    // DirectX staging texture for CPU readback
    ComPtr<ID3D11Texture2D> staging_texture_;
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
};