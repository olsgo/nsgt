import 'package:flutter/foundation.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:typed_data';
import '../widgets/spectrogram_canvas.dart';

class DSPServiceClient extends ChangeNotifier {
  static const String _defaultHost = 'localhost';
  static const int _defaultPort = 8000;
  
  WebSocketChannel? _channel;
  bool _isConnected = false;
  bool _isProcessing = false;
  bool _gpuAvailable = false;
  TextureInfo? _currentTexture;
  
  // Getters
  bool get isConnected => _isConnected;
  bool get isProcessing => _isProcessing;
  bool get gpuAvailable => _gpuAvailable;
  TextureInfo? get currentTexture => _currentTexture;
  
  // Connection management
  Future<void> connect({String host = _defaultHost, int port = _defaultPort}) async {
    try {
      // First check health endpoint
      final healthResponse = await http.get(
        Uri.parse('http://$host:$port/health'),
        headers: {'Content-Type': 'application/json'},
      );
      
      if (healthResponse.statusCode == 200) {
        final healthData = json.decode(healthResponse.body);
        _gpuAvailable = healthData['cuda_available'] ?? false;
        
        // Connect to WebSocket
        _channel = WebSocketChannel.connect(
          Uri.parse('ws://$host:$port/ws'),
        );
        
        _channel!.stream.listen(
          _handleMessage,
          onError: _handleError,
          onDone: _handleDisconnect,
        );
        
        _isConnected = true;
        notifyListeners();
        
        debugPrint('Connected to DSP service at $host:$port');
        debugPrint('GPU available: $_gpuAvailable');
      } else {
        throw Exception('DSP service health check failed');
      }
    } catch (e) {
      debugPrint('Failed to connect to DSP service: $e');
      _isConnected = false;
      notifyListeners();
    }
  }
  
  void disconnect() {
    _channel?.sink.close();
    _channel = null;
    _isConnected = false;
    _currentTexture = null;
    notifyListeners();
  }
  
  // Message handling
  void _handleMessage(dynamic message) {
    try {
      final data = json.decode(message);
      
      if (data['success'] == true) {
        if (data.containsKey('texture_id')) {
          // Texture update
          _currentTexture = TextureInfo(
            width: data['dimensions']['width'],
            height: data['dimensions']['height'],
            format: data['format'],
            textureId: data['texture_id'],
          );
          notifyListeners();
        }
      } else {
        debugPrint('DSP service error: ${data['error']}');
      }
      
      _isProcessing = false;
      notifyListeners();
    } catch (e) {
      debugPrint('Error parsing message: $e');
    }
  }
  
  void _handleError(error) {
    debugPrint('WebSocket error: $error');
    _isConnected = false;
    notifyListeners();
  }
  
  void _handleDisconnect() {
    debugPrint('WebSocket disconnected');
    _isConnected = false;
    notifyListeners();
  }
  
  // DSP operations
  Future<void> generateSpectrogram(List<double> audioData, String transformType) async {
    if (!_isConnected || _channel == null) return;
    
    _isProcessing = true;
    notifyListeners();
    
    final message = json.encode({
      'command': 'generate_spectrogram',
      'audio_data': audioData,
      'transform_type': transformType,
    });
    
    _channel!.sink.add(message);
  }
  
  Future<void> applyBrush(String brushType, Map<String, int> roi, Map<String, double> params) async {
    if (!_isConnected || _channel == null) return;
    
    final message = json.encode({
      'command': 'apply_brush',
      'brush_type': brushType,
      'roi': roi,
      'params': params,
    });
    
    _channel!.sink.add(message);
  }
  
  Future<void> reset() async {
    if (!_isConnected || _channel == null) return;
    
    _isProcessing = true;
    notifyListeners();
    
    final message = json.encode({
      'command': 'reset',
    });
    
    _channel!.sink.add(message);
  }
  
  // High-level operations
  Future<void> loadSampleAudio(String sampleType) async {
    // Generate synthetic audio data for demo
    List<double> audioData;
    
    switch (sampleType) {
      case 'piano':
        audioData = _generatePianoSample();
        break;
      case 'drums':
        audioData = _generateDrumSample();
        break;
      case 'voice':
        audioData = _generateVoiceSample();
        break;
      default:
        audioData = _generateSineSample();
    }
    
    await generateSpectrogram(audioData, 'nsgt');
  }
  
  Future<void> loadAudioFile() async {
    // TODO: Implement file picker integration
    debugPrint('File picker not implemented yet');
    
    // For now, load a default sample
    await loadSampleAudio('piano');
  }
  
  Future<void> regenerateSpectrogram(String transformType) async {
    if (_currentTexture == null) return;
    
    // For demo purposes, just simulate regeneration
    _isProcessing = true;
    notifyListeners();
    
    // Simulate processing delay
    await Future.delayed(const Duration(milliseconds: 500));
    
    _isProcessing = false;
    notifyListeners();
  }
  
  Future<void> exportImage() async {
    // TODO: Implement image export
    debugPrint('Image export not implemented yet');
  }
  
  Future<void> exportAudio() async {
    // TODO: Implement audio export
    debugPrint('Audio export not implemented yet');
  }
  
  // Sample audio generation
  List<double> _generatePianoSample() {
    const int sampleRate = 44100;
    const double duration = 2.0;
    final int numSamples = (sampleRate * duration).toInt();
    
    final List<double> samples = [];
    
    // Generate a simple piano-like sound (sum of harmonics)
    for (int i = 0; i < numSamples; i++) {
      final double t = i / sampleRate;
      double sample = 0.0;
      
      // Fundamental frequency (A4 = 440 Hz)
      sample += 0.5 * math.sin(2 * math.pi * 440 * t);
      
      // Add harmonics
      sample += 0.3 * math.sin(2 * math.pi * 880 * t);  // Octave
      sample += 0.2 * math.sin(2 * math.pi * 1320 * t); // Fifth
      sample += 0.1 * math.sin(2 * math.pi * 1760 * t); // Major third
      
      // Apply envelope (attack-decay)
      final double envelope = math.exp(-t * 2.0);
      sample *= envelope;
      
      samples.add(sample);
    }
    
    return samples;
  }
  
  List<double> _generateDrumSample() {
    const int sampleRate = 44100;
    const double duration = 1.5;
    final int numSamples = (sampleRate * duration).toInt();
    
    final List<double> samples = [];
    
    // Generate drum-like transients
    for (int i = 0; i < numSamples; i++) {
      final double t = i / sampleRate;
      double sample = 0.0;
      
      // Kick drum at t=0, 0.5, 1.0
      for (double kickTime in [0.0, 0.5, 1.0]) {
        if (t >= kickTime && t < kickTime + 0.1) {
          final double dt = t - kickTime;
          sample += 0.8 * math.sin(2 * math.pi * 60 * dt) * math.exp(-dt * 20);
        }
      }
      
      // Hi-hat at regular intervals
      for (double hatTime = 0.25; hatTime < duration; hatTime += 0.25) {
        if (t >= hatTime && t < hatTime + 0.05) {
          final double dt = t - hatTime;
          // White noise-like sound
          sample += 0.3 * (math.Random().nextDouble() * 2 - 1) * math.exp(-dt * 50);
        }
      }
      
      samples.add(sample);
    }
    
    return samples;
  }
  
  List<double> _generateVoiceSample() {
    const int sampleRate = 44100;
    const double duration = 1.8;
    final int numSamples = (sampleRate * duration).toInt();
    
    final List<double> samples = [];
    
    // Generate voice-like formant structure
    for (int i = 0; i < numSamples; i++) {
      final double t = i / sampleRate;
      double sample = 0.0;
      
      // Fundamental frequency varies (100-150 Hz)
      final double f0 = 120 + 30 * math.sin(2 * math.pi * 2 * t);
      
      // Add formants (vocal tract resonances)
      sample += 0.4 * math.sin(2 * math.pi * f0 * t);        // F0
      sample += 0.3 * math.sin(2 * math.pi * 800 * t);       // F1
      sample += 0.2 * math.sin(2 * math.pi * 1200 * t);      // F2
      sample += 0.1 * math.sin(2 * math.pi * 2600 * t);      // F3
      
      // Apply envelope
      final double envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * t);
      sample *= envelope;
      
      samples.add(sample);
    }
    
    return samples;
  }
  
  List<double> _generateSineSample() {
    const int sampleRate = 44100;
    const double duration = 2.0;
    final int numSamples = (sampleRate * duration).toInt();
    
    final List<double> samples = [];
    
    for (int i = 0; i < numSamples; i++) {
      final double t = i / sampleRate;
      final double sample = 0.5 * math.sin(2 * math.pi * 440 * t);
      samples.add(sample);
    }
    
    return samples;
  }
}

// Import math library
import 'dart:math' as math;