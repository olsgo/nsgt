import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import '../main.dart';
import '../platform/dsp_service_client.dart';

class SpectrogramCanvas extends StatefulWidget {
  const SpectrogramCanvas({super.key});

  @override
  State<SpectrogramCanvas> createState() => _SpectrogramCanvasState();
}

class _SpectrogramCanvasState extends State<SpectrogramCanvas> {
  FragmentShader? _blurShader;
  FragmentShader? _warpShader;
  bool _isDragging = false;
  Offset? _lastPanPosition;

  @override
  void initState() {
    super.initState();
    _loadShaders();
  }

  Future<void> _loadShaders() async {
    try {
      // Load fragment shaders
      final blurProgram = await FragmentProgram.fromAsset('shaders/blur.frag');
      final warpProgram = await FragmentProgram.fromAsset('shaders/warp.frag');
      
      setState(() {
        _blurShader = blurProgram.fragmentShader();
        _warpShader = warpProgram.fragmentShader();
      });
    } catch (e) {
      debugPrint('Error loading shaders: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Consumer2<DSPServiceClient, BrushState>(
      builder: (context, dspClient, brushState, child) {
        return GestureDetector(
          onPanStart: _onPanStart,
          onPanUpdate: (details) => _onPanUpdate(details, dspClient, brushState),
          onPanEnd: _onPanEnd,
          child: CustomPaint(
            painter: SpectrogramPainter(
              texture: dspClient.currentTexture,
              blurShader: _blurShader,
              warpShader: _warpShader,
              brushState: brushState,
              isDragging: _isDragging,
              brushPosition: _lastPanPosition,
            ),
            size: Size.infinite,
          ),
        );
      },
    );
  }

  void _onPanStart(DragStartDetails details) {
    setState(() {
      _isDragging = true;
      _lastPanPosition = details.localPosition;
    });
  }

  void _onPanUpdate(DragUpdateDetails details, DSPServiceClient dspClient, BrushState brushState) {
    setState(() {
      _lastPanPosition = details.localPosition;
    });

    // Apply brush effect through DSP service
    final renderBox = context.findRenderObject() as RenderBox?;
    if (renderBox != null && dspClient.currentTexture != null) {
      final size = renderBox.size;
      final texture = dspClient.currentTexture!;
      
      // Convert screen coordinates to texture coordinates
      final normalizedX = details.localPosition.dx / size.width;
      final normalizedY = details.localPosition.dy / size.height;
      
      // Calculate brush region in texture space
      final brushRadiusNormalized = brushState.brushRadius / size.width;
      final x = ((normalizedX - brushRadiusNormalized) * texture.width).clamp(0, texture.width).toInt();
      final y = ((normalizedY - brushRadiusNormalized) * texture.height).clamp(0, texture.height).toInt();
      final width = (brushRadiusNormalized * 2 * texture.width).clamp(1, texture.width - x).toInt();
      final height = (brushRadiusNormalized * 2 * texture.height).clamp(1, texture.height - y).toInt();
      
      // Send brush command to DSP service
      dspClient.applyBrush(
        brushState.currentBrush,
        {'x': x, 'y': y, 'width': width, 'height': height},
        {
          'radius': brushState.blurRadius,
          'strength': brushState.brushStrength,
          'displacement': brushState.displacement,
        },
      );
    }
  }

  void _onPanEnd(DragEndDetails details) {
    setState(() {
      _isDragging = false;
      _lastPanPosition = null;
    });
  }
}

class SpectrogramPainter extends CustomPainter {
  final TextureInfo? texture;
  final FragmentShader? blurShader;
  final FragmentShader? warpShader;
  final BrushState brushState;
  final bool isDragging;
  final Offset? brushPosition;

  SpectrogramPainter({
    required this.texture,
    required this.blurShader,
    required this.warpShader,
    required this.brushState,
    required this.isDragging,
    required this.brushPosition,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (texture == null) {
      // Draw placeholder
      _drawPlaceholder(canvas, size);
      return;
    }

    // Draw spectrogram texture
    _drawSpectrogram(canvas, size);

    // Draw brush overlay if dragging
    if (isDragging && brushPosition != null) {
      _drawBrushOverlay(canvas, size);
    }
  }

  void _drawPlaceholder(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.grey[800]!
      ..style = PaintingStyle.fill;

    canvas.drawRect(Offset.zero & size, paint);

    // Draw loading text
    final textStyle = TextStyle(
      color: Colors.white70,
      fontSize: 16,
      fontFamily: 'Roboto Mono',
    );

    final textSpan = TextSpan(
      text: 'Load an audio file to begin editing',
      style: textStyle,
    );

    final textPainter = TextPainter(
      text: textSpan,
      textDirection: TextDirection.ltr,
    );

    textPainter.layout();
    final textOffset = Offset(
      (size.width - textPainter.width) / 2,
      (size.height - textPainter.height) / 2,
    );

    textPainter.paint(canvas, textOffset);
  }

  void _drawSpectrogram(Canvas canvas, Size size) {
    if (texture == null) return;

    // For now, draw a placeholder rectangle representing the texture
    // In a real implementation, this would render the actual GPU texture
    final paint = Paint()
      ..color = Colors.blue[900]!
      ..style = PaintingStyle.fill;

    canvas.drawRect(Offset.zero & size, paint);

    // Draw grid lines to simulate spectrogram
    final gridPaint = Paint()
      ..color = Colors.white10
      ..strokeWidth = 1;

    // Frequency lines (horizontal)
    for (int i = 0; i < 10; i++) {
      final y = (i / 9) * size.height;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), gridPaint);
    }

    // Time lines (vertical)
    for (int i = 0; i < 20; i++) {
      final x = (i / 19) * size.width;
      canvas.drawLine(Offset(x, 0), Offset(x, size.height), gridPaint);
    }

    // Add some example spectral content
    final contentPaint = Paint()
      ..color = Colors.yellow.withOpacity(0.3)
      ..style = PaintingStyle.fill;

    for (int i = 0; i < 50; i++) {
      final x = (i / 49) * size.width;
      final y = size.height * 0.7 + 
                 (size.height * 0.2) * (0.5 + 0.5 * (i % 3 - 1) / 3);
      final rect = Rect.fromCenter(
        center: Offset(x, y),
        width: 4,
        height: 8,
      );
      canvas.drawRect(rect, contentPaint);
    }
  }

  void _drawBrushOverlay(Canvas canvas, Size size) {
    if (brushPosition == null) return;

    final paint = Paint()
      ..color = Colors.white.withOpacity(0.3)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    // Draw brush circle
    canvas.drawCircle(
      brushPosition!,
      brushState.brushRadius,
      paint,
    );

    // Draw brush type indicator
    final centerPaint = Paint()
      ..color = Colors.white.withOpacity(0.6)
      ..style = PaintingStyle.fill;

    if (brushState.currentBrush == 'blur') {
      // Draw blur indicator (filled circle)
      canvas.drawCircle(brushPosition!, 3, centerPaint);
    } else if (brushState.currentBrush == 'warp') {
      // Draw warp indicator (arrow)
      final path = Path();
      path.moveTo(brushPosition!.dx - 5, brushPosition!.dy);
      path.lineTo(brushPosition!.dx + 5, brushPosition!.dy);
      path.moveTo(brushPosition!.dx + 2, brushPosition!.dy - 3);
      path.lineTo(brushPosition!.dx + 5, brushPosition!.dy);
      path.lineTo(brushPosition!.dx + 2, brushPosition!.dy + 3);
      
      final arrowPaint = Paint()
        ..color = Colors.white.withOpacity(0.8)
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke
        ..strokeCap = StrokeCap.round;
      
      canvas.drawPath(path, arrowPaint);
    }
  }

  @override
  bool shouldRepaint(covariant SpectrogramPainter oldDelegate) {
    return texture != oldDelegate.texture ||
           isDragging != oldDelegate.isDragging ||
           brushPosition != oldDelegate.brushPosition ||
           brushState != oldDelegate.brushState;
  }
}

class TextureInfo {
  final int width;
  final int height;
  final String format;
  final String textureId;

  TextureInfo({
    required this.width,
    required this.height,
    required this.format,
    required this.textureId,
  });
}