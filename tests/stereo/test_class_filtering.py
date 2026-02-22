# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Unit tests for class filtering and reindexing functionality."""


class TestClassMappingUtilities:
    """Test suite for class mapping utility functions (T091)."""

    def test_get_paper_class_mapping(self):
        """Test get_paper_class_mapping returns correct mapping dictionaries."""
        from ultralytics.models.yolo.stereo3ddet.utils import (
            ORIGINAL_TO_PAPER,
            PAPER_TO_ORIGINAL,
            get_paper_class_mapping,
        )

        original_to_paper, paper_to_original = get_paper_class_mapping()

        # Verify mapping structure
        assert isinstance(original_to_paper, dict)
        assert isinstance(paper_to_original, dict)

        # Verify expected mappings
        assert original_to_paper[0] == 0  # Car -> Car
        assert original_to_paper[3] == 1  # Pedestrian -> Pedestrian
        assert original_to_paper[5] == 2  # Cyclist -> Cyclist

        # Verify reverse mapping
        assert paper_to_original[0] == 0  # Car -> Car
        assert paper_to_original[1] == 3  # Pedestrian -> Pedestrian
        assert paper_to_original[2] == 5  # Cyclist -> Cyclist

        # Verify constants match
        assert original_to_paper == ORIGINAL_TO_PAPER
        assert paper_to_original == PAPER_TO_ORIGINAL

    def test_filter_and_remap_class_id_valid(self):
        """Test filter_and_remap_class_id with valid class IDs."""
        from ultralytics.models.yolo.stereo3ddet.utils import filter_and_remap_class_id

        # Test valid mappings
        assert filter_and_remap_class_id(0) == 0  # Car -> Car
        assert filter_and_remap_class_id(3) == 1  # Pedestrian -> Pedestrian
        assert filter_and_remap_class_id(5) == 2  # Cyclist -> Cyclist

    def test_filter_and_remap_class_id_invalid(self):
        """Test filter_and_remap_class_id with invalid class IDs (should return None)."""
        from ultralytics.models.yolo.stereo3ddet.utils import filter_and_remap_class_id

        # Test invalid class IDs (should return None)
        assert filter_and_remap_class_id(1) is None  # Van - filtered out
        assert filter_and_remap_class_id(2) is None  # Truck - filtered out
        assert filter_and_remap_class_id(4) is None  # Person_sitting - filtered out
        assert filter_and_remap_class_id(6) is None  # Tram - filtered out
        assert filter_and_remap_class_id(7) is None  # Misc - filtered out
        assert filter_and_remap_class_id(99) is None  # Invalid ID
