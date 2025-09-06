import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../main.dart';

class BrushControls extends StatelessWidget {
  const BrushControls({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer<BrushState>(
      builder: (context, brushState, child) {
        return Container(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Brush Tools',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 16),
              
              // Brush type selection
              _buildBrushTypeSelector(context, brushState),
              const SizedBox(height: 16),
              
              // Common brush parameters
              _buildSlider(
                context,
                'Brush Radius',
                brushState.brushRadius,
                5.0,
                100.0,
                (value) => brushState.setBrushRadius(value),
                unit: 'px',
              ),
              
              _buildSlider(
                context,
                'Strength',
                brushState.brushStrength,
                0.0,
                1.0,
                (value) => brushState.setBrushStrength(value),
                unit: '',
              ),
              
              const SizedBox(height: 16),
              
              // Brush-specific parameters
              if (brushState.currentBrush == 'blur') ...[
                Text(
                  'Blur Parameters',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 8),
                _buildSlider(
                  context,
                  'Blur Radius',
                  brushState.blurRadius,
                  1.0,
                  20.0,
                  (value) => brushState.setBlurRadius(value),
                  unit: 'px',
                ),
              ],
              
              if (brushState.currentBrush == 'warp') ...[
                Text(
                  'Warp Parameters',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 8),
                _buildSlider(
                  context,
                  'Displacement',
                  brushState.displacement,
                  0.5,
                  10.0,
                  (value) => brushState.setDisplacement(value),
                  unit: 'px',
                ),
              ],
            ],
          ),
        );
      },
    );
  }

  Widget _buildBrushTypeSelector(BuildContext context, BrushState brushState) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Brush Type',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildBrushButton(
                context,
                'Blur',
                'blur',
                Icons.blur_on,
                brushState.currentBrush == 'blur',
                () => brushState.setBrush('blur'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildBrushButton(
                context,
                'Warp',
                'warp',
                Icons.gesture,
                brushState.currentBrush == 'warp',
                () => brushState.setBrush('warp'),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildBrushButton(
                context,
                'Enhance',
                'enhance',
                Icons.auto_fix_high,
                brushState.currentBrush == 'enhance',
                () => brushState.setBrush('enhance'),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildBrushButton(
                context,
                'Filter',
                'filter',
                Icons.tune,
                brushState.currentBrush == 'filter',
                () => brushState.setBrush('filter'),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildBrushButton(
    BuildContext context,
    String label,
    String value,
    IconData icon,
    bool isSelected,
    VoidCallback onPressed,
  ) {
    return ElevatedButton.icon(
      onPressed: onPressed,
      icon: Icon(icon, size: 16),
      label: Text(
        label,
        style: const TextStyle(fontSize: 12),
      ),
      style: ElevatedButton.styleFrom(
        backgroundColor: isSelected
            ? Theme.of(context).colorScheme.primary
            : Theme.of(context).colorScheme.surface,
        foregroundColor: isSelected
            ? Theme.of(context).colorScheme.onPrimary
            : Theme.of(context).colorScheme.onSurface,
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        minimumSize: const Size(0, 32),
      ),
    );
  }

  Widget _buildSlider(
    BuildContext context,
    String label,
    double value,
    double min,
    double max,
    ValueChanged<double> onChanged, {
    String unit = '',
    int divisions = 100,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label),
            Text(
              '${value.toStringAsFixed(1)}$unit',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                fontFamily: 'Roboto Mono',
              ),
            ),
          ],
        ),
        Slider(
          value: value,
          min: min,
          max: max,
          divisions: divisions,
          onChanged: onChanged,
        ),
      ],
    );
  }
}