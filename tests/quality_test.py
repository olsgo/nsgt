import numpy as np
import unittest
from nsgt import NSGT_sliced, OctScale, quality_check, comprehensive_quality_report


class TestQualityMeasurement(unittest.TestCase):
    """Test quality measurement framework for NSGT transforms"""
    
    def setUp(self):
        # Create test signal
        np.random.seed(42)
        self.sr = 44100
        self.sig_len = 20000
        t = np.linspace(0, self.sig_len/self.sr, self.sig_len, endpoint=False)
        
        # Multi-component test signal
        self.signal = (0.5 * np.sin(2*np.pi*440*t) + 
                      0.3 * np.sin(2*np.pi*880*t) +
                      0.1 * np.random.normal(0, 0.1, len(t)))
        self.signal = self.signal.astype(np.float32)

    def test_perfect_reconstruction_quality(self):
        """Test quality measurement with perfect reconstruction"""
        scale = OctScale(fmin=100, fmax=self.sr//2, bpo=12)
        nsgt = NSGT_sliced(scale, fs=self.sr, sl_len=8192, tr_area=1024)
        
        # Forward and backward transform
        coeffs = nsgt.forward((self.signal,))
        reconstructed = nsgt.backward(coeffs)
        s_r = np.concatenate(list(map(list, reconstructed)))[:len(self.signal)]
        
        # Test quality check
        passed, report = quality_check(self.signal, s_r, tolerance=1e-3, verbose=False)
        
        # Assertions
        self.assertTrue(passed, "Perfect reconstruction should pass quality check")
        self.assertGreater(report['sdr_db'], 60, "SDR should be > 60 dB for perfect reconstruction")
        self.assertLess(report['spectral_convergence'], 0.01, "Spectral convergence should be < 0.01")
        self.assertLess(report['rel_l2_error'], 0.01, "Relative L2 error should be < 0.01")

    def test_degraded_signal_quality(self):
        """Test quality measurement with artificially degraded signal"""
        # Add noise to simulate degradation
        noise_level = 0.1
        degraded = self.signal + noise_level * np.random.normal(0, 1, len(self.signal))
        
        passed, report = quality_check(self.signal, degraded, tolerance=1e-3, verbose=False)
        
        # Should fail strict tolerance but still provide meaningful metrics
        self.assertFalse(passed, "Degraded signal should fail strict quality check")
        self.assertLess(report['sdr_db'], 30, "SDR should be lower for degraded signal")
        self.assertGreater(report['spectral_convergence'], 0.01, "Spectral convergence should be higher")

    def test_individual_metrics(self):
        """Test individual quality metrics"""
        from nsgt.quality import sdr, sir, sar, spectral_convergence
        
        # Perfect copy should have infinite SDR
        perfect_sdr = sdr(self.signal, self.signal)
        self.assertEqual(perfect_sdr, float('inf'), "Perfect reconstruction should have infinite SDR")
        
        # Scaled signal should have measurable but good SDR
        scaled = 0.9 * self.signal
        scaled_sdr = sdr(self.signal, scaled)
        self.assertGreater(scaled_sdr, 19, "Scaled signal should have reasonable SDR")
        
        # Test spectral convergence
        conv = spectral_convergence(self.signal, scaled)
        self.assertGreater(conv, 0, "Spectral convergence should be positive")
        self.assertLess(conv, 1, "Spectral convergence should be reasonable")

    def test_windowed_sdr(self):
        """Test SDR calculation with different windows"""
        from nsgt.quality import sdr
        
        # Test with different windows
        scaled = 0.95 * self.signal
        sdr_rect = sdr(self.signal, scaled)
        sdr_hann = sdr(self.signal, scaled, window='hann')
        sdr_hamming = sdr(self.signal, scaled, window='hamming')
        
        # All should be finite and positive
        for sdr_val in [sdr_rect, sdr_hann, sdr_hamming]:
            self.assertGreater(sdr_val, 0, "SDR should be positive")
            self.assertLess(sdr_val, float('inf'), "SDR should be finite")

    def test_comprehensive_report(self):
        """Test comprehensive quality report generation"""
        # Create slightly degraded version
        degraded = self.signal + 0.01 * np.random.normal(0, 1, len(self.signal))
        
        report = comprehensive_quality_report(self.signal, degraded)
        
        # Check all expected keys are present
        expected_keys = ['sdr_db', 'sir_db', 'sar_db', 'spectral_convergence',
                        'l1_error', 'l2_error', 'linf_error', 'rel_l2_error', 'snr_estimate_db']
        
        for key in expected_keys:
            self.assertIn(key, report, f"Report should contain {key}")
            self.assertIsInstance(report[key], (int, float), f"{key} should be numeric")
            self.assertFalse(np.isnan(report[key]), f"{key} should not be NaN")

    def test_error_metrics(self):
        """Test different error metric calculations"""
        from nsgt.quality import perfect_reconstruction_error
        
        # Create test signals with known error
        ref = np.array([1, 2, 3, 4, 5], dtype=float)
        est = np.array([1.1, 2.1, 3.1, 4.1, 5.1], dtype=float)  # +0.1 error
        
        l1_err = perfect_reconstruction_error(ref, est, 'l1')
        l2_err = perfect_reconstruction_error(ref, est, 'l2')
        linf_err = perfect_reconstruction_error(ref, est, 'linf')
        rel_l2_err = perfect_reconstruction_error(ref, est, 'rel_l2')
        
        self.assertAlmostEqual(l1_err, 0.5, places=10, msg="L1 error should be 0.5")
        self.assertAlmostEqual(linf_err, 0.1, places=10, msg="L∞ error should be 0.1")
        self.assertAlmostEqual(l2_err, np.sqrt(5*0.01), places=10, msg="L2 error should be √0.05")
        self.assertGreater(rel_l2_err, 0, msg="Relative L2 error should be positive")


if __name__ == '__main__':
    unittest.main()