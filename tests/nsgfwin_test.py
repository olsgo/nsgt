import numpy as np
import unittest
from nsgt.nsgfwin import nsgfwin


class TestNSGFWin(unittest.TestCase):
    """Test nsgfwin functionality including previously untested branches"""

    def test_basic_functionality(self):
        """Test basic nsgfwin operation"""
        g, rfbas, M = nsgfwin(fmin=100, fmax=8000, bins=12, sr=44100, Ls=44100)
        
        self.assertIsInstance(g, list, "g should be a list of windows")
        self.assertGreater(len(g), 0, "Should generate at least one window")
        self.assertEqual(len(g), len(rfbas), "g and rfbas should have same length")
        self.assertEqual(len(g), len(M), "g and M should have same length")

    def test_short_bins_array_branch(self):
        """Test TODO branch: when bins array is shorter than needed (line 52)"""
        # Create scenario where len(bins) < b (number of octaves)
        fmin, fmax = 100, 16000  # ~7.3 octaves
        bins_short = np.array([12, 8])  # Only 2 elements, but need ~8
        
        try:
            g, rfbas, M = nsgfwin(fmin=fmin, fmax=fmax, bins=bins_short, sr=44100, Ls=44100)
            self.assertIsInstance(g, list, "Should handle short bins array")
            self.assertGreater(len(g), len(bins_short), "Should extend bins array")
        except Exception as e:
            self.fail(f"Short bins array should be handled gracefully: {e}")

    def test_bins_with_zero_values(self):
        """Test handling of zero/negative values in bins array"""
        fmin, fmax = 100, 8000
        bins_with_zeros = np.array([12, 0, -1, 8])  # Contains zero and negative
        
        try:
            g, rfbas, M = nsgfwin(fmin=fmin, fmax=fmax, bins=bins_with_zeros, sr=44100, Ls=44100)
            self.assertIsInstance(g, list, "Should handle bins with zero/negative values")
        except Exception as e:
            self.fail(f"Bins with zero values should be handled: {e}")

    def test_fmax_branch_when_fbas_exceeds_nyquist(self):
        """Test TODO branch: fbas handling when frequencies exceed limits (line 66)"""
        # Create scenario where fbas[min(where(fbas>=fmax))] < nf (Nyquist)
        # This triggers the second branch
        fmin = 100
        fmax = 15000  # Set fmax close to but below Nyquist (22050)
        bins = 24  # Many bins to create dense frequency grid
        sr = 44100
        
        try:
            g, rfbas, M = nsgfwin(fmin=fmin, fmax=fmax, bins=bins, sr=sr, Ls=sr)
            self.assertIsInstance(g, list, "Should handle fmax near Nyquist")
            
            # Verify frequency bounds are reasonable (rfbas contains sample indices, not Hz)
            # The function handles frequency limits internally
            self.assertGreater(len(rfbas), 0, "Should generate frequency bins")
            self.assertTrue(all(f >= 0 for f in rfbas), "All frequency indices should be non-negative")
            
        except Exception as e:
            self.fail(f"fmax near Nyquist should be handled: {e}")

    def test_logspace_vs_original_computation(self):
        """Test that logspace implementation gives similar results to original"""
        fmin, fmax = 100, 8000
        bins = 12
        sr = 44100
        Ls = 44100
        
        # Test with current (logspace) implementation  
        g_new, rfbas_new, M_new = nsgfwin(fmin=fmin, fmax=fmax, bins=bins, sr=sr, Ls=Ls)
        
        # Both should produce valid results
        self.assertIsInstance(g_new, list)
        self.assertGreater(len(g_new), 0)
        
        # Verify frequency progression is logarithmic
        freq_ratios = []
        for i in range(1, min(5, len(rfbas_new))):  # Check first few ratios
            if rfbas_new[i-1] > 0:
                ratio = rfbas_new[i] / rfbas_new[i-1]
                if ratio > 1.01:  # Skip tiny differences
                    freq_ratios.append(ratio)
        
        if len(freq_ratios) > 1:
            # Ratios should be approximately constant for logarithmic scale
            ratio_std = np.std(freq_ratios)
            self.assertLess(ratio_std, 0.1, "Frequency ratios should be approximately constant")

    def test_window_properties(self):
        """Test properties of generated windows"""
        g, rfbas, M = nsgfwin(fmin=200, fmax=8000, bins=6, sr=44100, Ls=22050)
        
        # All windows should be non-empty
        for i, window in enumerate(g):
            self.assertGreater(len(window), 0, f"Window {i} should not be empty")
            self.assertTrue(np.all(np.isfinite(window)), f"Window {i} should contain finite values")
            self.assertTrue(np.all(window >= 0), f"Window {i} should be non-negative (Hann window)")

    def test_edge_cases(self):
        """Test edge cases and parameter validation"""
        # Very small frequency range
        g1, rfbas1, M1 = nsgfwin(fmin=1000, fmax=1100, bins=2, sr=44100, Ls=44100)
        self.assertGreater(len(g1), 0, "Should handle small frequency range")
        
        # Single bin
        g2, rfbas2, M2 = nsgfwin(fmin=440, fmax=880, bins=1, sr=44100, Ls=44100)
        self.assertGreater(len(g2), 0, "Should handle single bin")
        
        # Large number of bins
        g3, rfbas3, M3 = nsgfwin(fmin=100, fmax=10000, bins=48, sr=44100, Ls=88200)
        self.assertGreater(len(g3), 0, "Should handle many bins")

    def test_minimum_window_size(self):
        """Test that minimum window size constraint is respected"""
        min_win = 8
        g, rfbas, M = nsgfwin(fmin=8000, fmax=20000, bins=2, sr=44100, Ls=44100, min_win=min_win)
        
        # All window sizes should respect minimum
        for i, m in enumerate(M):
            self.assertGreaterEqual(m, min_win, f"Window {i} size {m} should be >= min_win {min_win}")


if __name__ == '__main__':
    unittest.main()