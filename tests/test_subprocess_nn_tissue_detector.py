import pytest
import numpy as np
from histpat_toolkit.tissue_detection import (
    NNTissueDetector,
    SubprocessNNTissueDetector,
)


class TestSubprocessNNTissueDetector:
    """Test suite for SubprocessNNTissueDetector"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample RGB image for testing"""
        # Create a simple test image with some variation
        # Using 512x512 to match the model's expected tile size
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        # Add a "tissue-like" region (darker pixels)
        img[100:400, 100:400, :] = img[100:400, 100:400, :] // 2
        return img
    
    @pytest.fixture
    def sample_image_with_alpha(self):
        """Create a sample RGBA image for testing"""
        img = np.random.randint(0, 255, (512, 512, 4), dtype=np.uint8)
        img[100:400, 100:400, :3] = img[100:400, 100:400, :3] // 2
        return img
    
    def test_initialization(self):
        """Test that SubprocessNNTissueDetector initializes correctly"""
        detector = SubprocessNNTissueDetector()
        assert detector.tile_size == 512
        assert detector.binary is True
        assert detector.mpp == 10
        
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters"""
        detector = SubprocessNNTissueDetector(
            tile_size=512,
            binary=False,
            opening_ksize=5,
            closing_ksize=7,
        )
        assert detector.tile_size == 512
        assert detector.binary is False
        assert detector.opening_ksize == 5
        assert detector.closing_ksize == 7
        
    def test_detect_tissue_basic(self, sample_image):
        """Test basic tissue detection on a sample image"""
        detector = SubprocessNNTissueDetector()  # Use default tile_size=512
        mask = detector._detect_tissue(sample_image)
        
        # Check output properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape == (sample_image.shape[0], sample_image.shape[1])
        assert set(np.unique(mask)).issubset({0, 1})  # Binary mask
        
    def test_detect_tissue_with_alpha_channel(self, sample_image_with_alpha):
        """Test that alpha channel is properly handled"""
        detector = SubprocessNNTissueDetector()  # Use default tile_size=512
        mask = detector._detect_tissue(sample_image_with_alpha)
        
        # Check output properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape == (sample_image_with_alpha.shape[0], sample_image_with_alpha.shape[1])
        
    def test_detect_tissue_non_binary(self, sample_image):
        """Test tissue detection with non-binary output"""
        detector = SubprocessNNTissueDetector(binary=False)  # Use default tile_size=512
        mask = detector._detect_tissue(sample_image)
        
        # Check output properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint8
        assert mask.shape == (sample_image.shape[0], sample_image.shape[1])
        # Non-binary mask can have values 0, 1, 2 (background, tissue, connective tissue)
        # But after postprocessing it has values from -1 to 2
        
    def test_consistency_with_regular_detector(self, sample_image):
        """Test that subprocess detector produces same results as regular detector"""
        # Create both detectors with same parameters
        regular_detector = NNTissueDetector(binary=True)  # Use default tile_size=512
        subprocess_detector = SubprocessNNTissueDetector(binary=True)
        
        # Run detection
        regular_mask = regular_detector._detect_tissue(sample_image)
        subprocess_mask = subprocess_detector._detect_tissue(sample_image)
        
        # Results should be identical
        assert regular_mask.shape == subprocess_mask.shape
        assert regular_mask.dtype == subprocess_mask.dtype
        np.testing.assert_array_equal(regular_mask, subprocess_mask)
        
    def test_consistency_non_binary(self, sample_image):
        """Test consistency for non-binary mode"""
        regular_detector = NNTissueDetector(binary=False)  # Use default tile_size=512
        subprocess_detector = SubprocessNNTissueDetector(binary=False)
        
        regular_mask = regular_detector._detect_tissue(sample_image)
        subprocess_mask = subprocess_detector._detect_tissue(sample_image)
        
        assert regular_mask.shape == subprocess_mask.shape
        assert regular_mask.dtype == subprocess_mask.dtype
        np.testing.assert_array_equal(regular_mask, subprocess_mask)
        
    def test_detect_tissue_public_method(self, sample_image):
        """Test the public detect_tissue method (with resizing support)"""
        detector = SubprocessNNTissueDetector()  # Use default tile_size=512
        
        # Test without mpp (no resizing)
        mask = detector.detect_tissue(sample_image)
        assert mask.shape == (sample_image.shape[0], sample_image.shape[1])
        
        # Test with mpp (should resize)
        mask_with_mpp = detector.detect_tissue(sample_image, img_mpp=5.0)
        # After resizing to mpp=10 and back, should have same shape
        assert mask_with_mpp.shape == (sample_image.shape[0], sample_image.shape[1])
        
    def test_different_image_sizes(self):
        """Test detection with different image sizes"""
        # Test with images of different sizes (all using default tile_size=512)
        detector = SubprocessNNTissueDetector()
        for size in [512, 1024]:
            test_image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            mask = detector._detect_tissue(test_image)
            assert mask.shape == (test_image.shape[0], test_image.shape[1])
            
    def test_large_image(self):
        """Test detection on a larger image"""
        large_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        detector = SubprocessNNTissueDetector(tile_size=512)
        mask = detector._detect_tissue(large_image)
        
        assert mask.shape == (large_image.shape[0], large_image.shape[1])
        assert mask.dtype == np.uint8
        
    def test_small_image(self):
        """Test detection on a very small image (smaller than tile_size)"""
        small_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        detector = SubprocessNNTissueDetector()  # tile_size=512
        mask = detector._detect_tissue(small_image)
        
        assert mask.shape == (small_image.shape[0], small_image.shape[1])
        assert mask.dtype == np.uint8
        
    def test_subprocess_error_handling(self, sample_image):
        """Test that subprocess errors are properly handled"""
        detector = SubprocessNNTissueDetector()
        
        # Temporarily break the worker script path to trigger an error
        original_path = detector._worker_script
        detector._worker_script = "/nonexistent/path/worker.py"
        
        with pytest.raises((RuntimeError, FileNotFoundError)):
            detector._detect_tissue(sample_image)
        
        # Restore the path
        detector._worker_script = original_path
        
    def test_multiple_detections(self, sample_image):
        """Test that detector can be used multiple times"""
        detector = SubprocessNNTissueDetector()
        
        # Run detection multiple times
        mask1 = detector._detect_tissue(sample_image)
        mask2 = detector._detect_tissue(sample_image)
        mask3 = detector._detect_tissue(sample_image)
        
        # All results should be identical
        np.testing.assert_array_equal(mask1, mask2)
        np.testing.assert_array_equal(mask2, mask3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
