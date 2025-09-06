import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../main.dart';
import '../platform/dsp_service_client.dart';

class TransformControls extends StatelessWidget {
  const TransformControls({super.key});

  @override
  Widget build(BuildContext context) {
    return Consumer2<SpectrogramState, DSPServiceClient>(
      builder: (context, spectrogramState, dspClient, child) {
        return Container(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Transform Settings',
                style: Theme.of(context).textTheme.titleMedium,
              ),
              const SizedBox(height: 16),
              
              // Transform type selection
              _buildTransformTypeSelector(context, spectrogramState, dspClient),
              const SizedBox(height: 16),
              
              // Display options
              _buildDisplayOptions(context, spectrogramState),
              const SizedBox(height: 16),
              
              // Action buttons
              _buildActionButtons(context, dspClient),
            ],
          ),
        );
      },
    );
  }

  Widget _buildTransformTypeSelector(
    BuildContext context,
    SpectrogramState state,
    DSPServiceClient dspClient,
  ) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Transform Type',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          value: state.transformType,
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          ),
          items: const [
            DropdownMenuItem(value: 'nsgt', child: Text('NSGT (Non-Stationary Gabor)')),
            DropdownMenuItem(value: 'stft', child: Text('STFT (Short-Time Fourier)')),
            DropdownMenuItem(value: 'mel', child: Text('Mel Spectrogram')),
            DropdownMenuItem(value: 'cqt', child: Text('CQT (Constant-Q)')),
            DropdownMenuItem(value: 'vqt', child: Text('VQT (Variable-Q)')),
          ],
          onChanged: (value) {
            if (value != null) {
              state.setTransformType(value);
              // Regenerate spectrogram with new transform
              dspClient.regenerateSpectrogram(value);
            }
          },
        ),
      ],
    );
  }

  Widget _buildDisplayOptions(BuildContext context, SpectrogramState state) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Display Options',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: 8),
        
        // Log scale toggle
        SwitchListTile(
          title: const Text('Log Scale'),
          subtitle: const Text('Use logarithmic amplitude scaling'),
          value: state.logScale,
          onChanged: (value) => state.setLogScale(value),
          dense: true,
          contentPadding: EdgeInsets.zero,
        ),
        
        // Colormap selection
        const SizedBox(height: 8),
        DropdownButtonFormField<String>(
          value: state.colormap,
          decoration: const InputDecoration(
            labelText: 'Colormap',
            border: OutlineInputBorder(),
            contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          ),
          items: const [
            DropdownMenuItem(value: 'viridis', child: Text('Viridis')),
            DropdownMenuItem(value: 'hot', child: Text('Hot')),
            DropdownMenuItem(value: 'cool', child: Text('Cool')),
            DropdownMenuItem(value: 'grayscale', child: Text('Grayscale')),
          ],
          onChanged: (value) {
            if (value != null) {
              state.setColormap(value);
            }
          },
        ),
      ],
    );
  }

  Widget _buildActionButtons(BuildContext context, DSPServiceClient dspClient) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        ElevatedButton.icon(
          onPressed: dspClient.isProcessing
              ? null
              : () => _showLoadAudioDialog(context, dspClient),
          icon: const Icon(Icons.audio_file),
          label: const Text('Load Audio'),
        ),
        const SizedBox(height: 8),
        ElevatedButton.icon(
          onPressed: dspClient.currentTexture == null
              ? null
              : () => dspClient.exportImage(),
          icon: const Icon(Icons.save),
          label: const Text('Export Image'),
        ),
        const SizedBox(height: 8),
        ElevatedButton.icon(
          onPressed: dspClient.currentTexture == null
              ? null
              : () => dspClient.exportAudio(),
          icon: const Icon(Icons.audiotrack),
          label: const Text('Export Audio'),
        ),
      ],
    );
  }

  void _showLoadAudioDialog(BuildContext context, DSPServiceClient dspClient) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Load Audio File'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Select an audio file to analyze:'),
            const SizedBox(height: 16),
            
            // Sample audio options for demo
            _buildSampleOption(
              context,
              'Piano Recording',
              'Sample piano piece for testing',
              () {
                Navigator.of(context).pop();
                dspClient.loadSampleAudio('piano');
              },
            ),
            const SizedBox(height: 8),
            _buildSampleOption(
              context,
              'Drum Loop',
              'Rhythmic content with percussive elements',
              () {
                Navigator.of(context).pop();
                dspClient.loadSampleAudio('drums');
              },
            ),
            const SizedBox(height: 8),
            _buildSampleOption(
              context,
              'Voice Recording',
              'Human speech for vocal analysis',
              () {
                Navigator.of(context).pop();
                dspClient.loadSampleAudio('voice');
              },
            ),
            
            const SizedBox(height: 16),
            const Divider(),
            const SizedBox(height: 8),
            
            ElevatedButton.icon(
              onPressed: () {
                Navigator.of(context).pop();
                // TODO: Implement file picker
                dspClient.loadAudioFile();
              },
              icon: const Icon(Icons.folder_open),
              label: const Text('Browse Files'),
            ),
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

  Widget _buildSampleOption(
    BuildContext context,
    String title,
    String subtitle,
    VoidCallback onTap,
  ) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(8),
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          border: Border.all(color: Theme.of(context).dividerColor),
          borderRadius: BorderRadius.circular(8),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}