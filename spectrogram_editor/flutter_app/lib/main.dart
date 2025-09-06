import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'widgets/spectrogram_canvas.dart';
import 'widgets/brush_controls.dart';
import 'widgets/transform_controls.dart';
import 'platform/dsp_service_client.dart';

void main() {
  runApp(const SpectrogramEditorApp());
}

class SpectrogramEditorApp extends StatelessWidget {
  const SpectrogramEditorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => DSPServiceClient()),
        ChangeNotifierProvider(create: (_) => BrushState()),
        ChangeNotifierProvider(create: (_) => SpectrogramState()),
      ],
      child: MaterialApp(
        title: 'Spectrogram Editor',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(
            seedColor: Colors.deepPurple,
            brightness: Brightness.dark,
          ),
          useMaterial3: true,
        ),
        home: const SpectrogramEditorHome(),
      ),
    );
  }
}

class SpectrogramEditorHome extends StatefulWidget {
  const SpectrogramEditorHome({super.key});

  @override
  State<SpectrogramEditorHome> createState() => _SpectrogramEditorHomeState();
}

class _SpectrogramEditorHomeState extends State<SpectrogramEditorHome> {
  @override
  void initState() {
    super.initState();
    // Initialize DSP service connection
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<DSPServiceClient>().connect();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: const Text('Spectrogram Editor'),
        actions: [
          IconButton(
            onPressed: () => _showLoadAudioDialog(context),
            icon: const Icon(Icons.audio_file),
            tooltip: 'Load Audio File',
          ),
          IconButton(
            onPressed: () => context.read<DSPServiceClient>().reset(),
            icon: const Icon(Icons.refresh),
            tooltip: 'Reset to Original',
          ),
        ],
      ),
      body: Row(
        children: [
          // Left panel - Controls
          SizedBox(
            width: 300,
            child: Column(
              children: [
                const TransformControls(),
                const Divider(),
                const BrushControls(),
                const Divider(),
                Expanded(
                  child: Container(
                    padding: const EdgeInsets.all(16),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Status',
                          style: Theme.of(context).textTheme.titleMedium,
                        ),
                        const SizedBox(height: 8),
                        Consumer<DSPServiceClient>(
                          builder: (context, client, child) {
                            return Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                _buildStatusItem(
                                  'Connection',
                                  client.isConnected ? 'Connected' : 'Disconnected',
                                  client.isConnected ? Colors.green : Colors.red,
                                ),
                                _buildStatusItem(
                                  'GPU Available',
                                  client.gpuAvailable ? 'Yes' : 'No',
                                  client.gpuAvailable ? Colors.green : Colors.orange,
                                ),
                                if (client.currentTexture != null)
                                  _buildStatusItem(
                                    'Texture',
                                    '${client.currentTexture!.width}×${client.currentTexture!.height}',
                                    Colors.blue,
                                  ),
                              ],
                            );
                          },
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
          const VerticalDivider(width: 1),
          
          // Main canvas area
          Expanded(
            child: Container(
              color: Colors.black,
              child: const SpectrogramCanvas(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusItem(String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          Container(
            width: 8,
            height: 8,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
            ),
          ),
          const SizedBox(width: 8),
          Text('$label: $value'),
        ],
      ),
    );
  }

  void _showLoadAudioDialog(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Load Audio File'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ElevatedButton.icon(
              onPressed: () {
                Navigator.of(context).pop();
                // TODO: Implement file picker
              },
              icon: const Icon(Icons.file_open),
              label: const Text('Browse Files'),
            ),
            const SizedBox(height: 16),
            const Text('Supported formats: WAV, MP3, FLAC, OGG'),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }
}

// State classes for Provider
class BrushState extends ChangeNotifier {
  String _currentBrush = 'blur';
  double _brushRadius = 20.0;
  double _brushStrength = 0.5;
  double _blurRadius = 5.0;
  double _displacement = 2.0;

  String get currentBrush => _currentBrush;
  double get brushRadius => _brushRadius;
  double get brushStrength => _brushStrength;
  double get blurRadius => _blurRadius;
  double get displacement => _displacement;

  void setBrush(String brush) {
    _currentBrush = brush;
    notifyListeners();
  }

  void setBrushRadius(double radius) {
    _brushRadius = radius;
    notifyListeners();
  }

  void setBrushStrength(double strength) {
    _brushStrength = strength;
    notifyListeners();
  }

  void setBlurRadius(double radius) {
    _blurRadius = radius;
    notifyListeners();
  }

  void setDisplacement(double displacement) {
    _displacement = displacement;
    notifyListeners();
  }
}

class SpectrogramState extends ChangeNotifier {
  String _transformType = 'nsgt';
  bool _logScale = true;
  String _colormap = 'viridis';

  String get transformType => _transformType;
  bool get logScale => _logScale;
  String get colormap => _colormap;

  void setTransformType(String type) {
    _transformType = type;
    notifyListeners();
  }

  void setLogScale(bool logScale) {
    _logScale = logScale;
    notifyListeners();
  }

  void setColormap(String colormap) {
    _colormap = colormap;
    notifyListeners();
  }
}