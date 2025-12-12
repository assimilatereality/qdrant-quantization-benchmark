"""
Tests for data upload operations with retry logic.
"""

import pytest
import time
from unittest.mock import Mock
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import PointStruct

from qdrant_quantization_benchmark.uploader import DataUploader
from qdrant_quantization_benchmark.config import UploadConfig


class TestDataUploader:
    """Tests for DataUploader class."""
    
    def test_initialization(self, mock_qdrant_client, upload_config):
        """Test uploader initialization."""
        uploader = DataUploader(mock_qdrant_client, upload_config)
        
        assert uploader.client == mock_qdrant_client
        assert uploader.config == upload_config
    
    def test_initialization_with_default_config(self, mock_qdrant_client):
        """Test initialization with default config."""
        uploader = DataUploader(mock_qdrant_client)
        
        assert isinstance(uploader.config, UploadConfig)
        assert uploader.config.batch_size == 50
    
    def test_upload_batch_simple(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test simple batch upload without retry."""
        config = UploadConfig(batch_size=50, enable_retry=False)
        uploader = DataUploader(mock_qdrant_client, config)
        
        result = uploader.upload_batch(
            collection_name="test",
            dataset=sample_dataset,
            embeddings=sample_embeddings,
            show_progress=False
        )
        
        assert result == len(sample_dataset)
        assert mock_qdrant_client.upsert.called
    
    def test_upload_batch_size_mismatch_raises_error(
        self, mock_qdrant_client, sample_dataset
    ):
        """Test that mismatched dataset and embeddings sizes raise error."""
        uploader = DataUploader(mock_qdrant_client)
        
        wrong_size_embeddings = [[0.1] * 384]  # Only 1 embedding
        
        with pytest.raises(ValueError, match="doesn't match"):
            uploader.upload_batch(
                collection_name="test",
                dataset=sample_dataset,  # 20 items
                embeddings=wrong_size_embeddings,  # 1 item
                show_progress=False
            )
    
    def test_upload_batch_small_batches(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test upload with small batch size."""
        config = UploadConfig(batch_size=5)  # Very small batches
        uploader = DataUploader(mock_qdrant_client, config)
        
        result = uploader.upload_batch(
            collection_name="test",
            dataset=sample_dataset,  # 20 items
            embeddings=sample_embeddings,
            show_progress=False
        )
        
        assert result == 20
        # Should be called 4 times (20 items / 5 batch_size)
        assert mock_qdrant_client.upsert.call_count == 4
    
    def test_upload_batch_named_vectors(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test upload with named vectors."""
        uploader = DataUploader(mock_qdrant_client)
        
        uploader.upload_batch(
            collection_name="test",
            dataset=sample_dataset,
            embeddings=sample_embeddings,
            named_vector=True,
            vector_name="dense",
            show_progress=False
        )
        
        # Check that points were created with named vectors
        call_args = mock_qdrant_client.upsert.call_args_list[0]
        points = call_args[1]['points']
        
        # Check first point has named vector
        assert isinstance(points[0].vector, dict)
        assert "dense" in points[0].vector
    
    def test_upload_batch_unnamed_vectors(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test upload with unnamed vectors."""
        uploader = DataUploader(mock_qdrant_client)
        
        uploader.upload_batch(
            collection_name="test",
            dataset=sample_dataset,
            embeddings=sample_embeddings,
            named_vector=False,
            show_progress=False
        )
        
        # Check that points were created with unnamed vectors
        call_args = mock_qdrant_client.upsert.call_args_list[0]
        points = call_args[1]['points']
        
        # Check first point has list vector
        assert isinstance(points[0].vector, list)
    
    def test_upload_batch_progress(self, mock_qdrant_client, capsys):
        """Test progress output during upload."""
        # Create larger dataset to trigger progress
        dataset = [{"id": i, "title": f"Item {i}", "description": f"Desc {i}"} for i in range(1500)]
        embeddings = [[0.1] * 384 for _ in range(1500)]
        
        config = UploadConfig(batch_size=50)
        uploader = DataUploader(mock_qdrant_client, config)
        
        uploader.upload_batch(
            collection_name="test",
            dataset=dataset,
            embeddings=embeddings,
            show_progress=True
        )
        
        captured = capsys.readouterr()
        assert "Progress:" in captured.out
        assert "1000" in captured.out  # Should show 1000 milestone
    
    def test_prepare_points(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test _prepare_points creates correct PointStruct objects."""
        uploader = DataUploader(mock_qdrant_client)
        
        points = uploader._prepare_points(
            batch_dataset=sample_dataset[:5],
            batch_embeddings=sample_embeddings[:5],
            start_id=0,
            named_vector=True,
            vector_name="dense"
        )
        
        assert len(points) == 5
        assert all(isinstance(p, PointStruct) for p in points)
        assert points[0].id == 0
        assert points[4].id == 4
        assert all(isinstance(p.vector, dict) for p in points)
    
    def test_prepare_points_with_offset(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test _prepare_points with start_id offset."""
        uploader = DataUploader(mock_qdrant_client)
        
        points = uploader._prepare_points(
            batch_dataset=sample_dataset[:5],
            batch_embeddings=sample_embeddings[:5],
            start_id=100,
            named_vector=False,
            vector_name="dense"
        )
        
        assert points[0].id == 100
        assert points[4].id == 104
    
    def test_upload_with_retry_disabled(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test that retry is not used when disabled."""
        config = UploadConfig(enable_retry=False)
        uploader = DataUploader(mock_qdrant_client, config)
        
        uploader.upload_batch(
            collection_name="test",
            dataset=sample_dataset,
            embeddings=sample_embeddings,
            show_progress=False
        )
        
        # Should use simple upsert, not retry method
        mock_qdrant_client.upsert.assert_called()
    
    def test_upload_with_retry_enabled_success(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test retry logic when enabled and upload succeeds."""
        config = UploadConfig(enable_retry=True, batch_size=50)
        uploader = DataUploader(mock_qdrant_client, config)
        
        result = uploader.upload_batch(
            collection_name="test",
            dataset=sample_dataset,
            embeddings=sample_embeddings,
            show_progress=False
        )
        
        assert result == len(sample_dataset)
        mock_qdrant_client.upsert.assert_called()
    
    def test_retry_succeeds_after_one_failure(self, mock_qdrant_client, capsys):
        """Test retry logic succeeds on second attempt."""
        config = UploadConfig(enable_retry=True, max_retries=3)
        uploader = DataUploader(mock_qdrant_client, config)
        
        # First call fails, second succeeds
        mock_qdrant_client.upsert.side_effect = [
            ResponseHandlingException("Timeout"),
            None  # Success
        ]
        
        points = [Mock(spec=PointStruct)]
        result = uploader._upload_with_retry("test", points, batch_num=0)
        
        assert result == 1
        assert mock_qdrant_client.upsert.call_count == 2
        
        captured = capsys.readouterr()
        assert "retrying" in captured.out.lower()
    
    def test_retry_fails_after_max_attempts(self, mock_qdrant_client):
        """Test retry logic fails after max retries."""
        config = UploadConfig(enable_retry=True, max_retries=2)
        uploader = DataUploader(mock_qdrant_client, config)
        
        # All attempts fail
        mock_qdrant_client.upsert.side_effect = ResponseHandlingException("Timeout")
        
        points = [Mock(spec=PointStruct)]
        
        with pytest.raises(ResponseHandlingException):
            uploader._upload_with_retry("test", points, batch_num=0)
        
        assert mock_qdrant_client.upsert.call_count == 2
    
    def test_retry_exponential_backoff(self, mock_qdrant_client, mocker):
        """Test that retry uses exponential backoff."""
        config = UploadConfig(enable_retry=True, max_retries=3, initial_backoff=1.0)
        uploader = DataUploader(mock_qdrant_client, config)
        
        # Mock time.sleep
        mock_sleep = mocker.patch('time.sleep')
        
        # First two calls fail, third succeeds
        mock_qdrant_client.upsert.side_effect = [
            ResponseHandlingException("Timeout"),
            ResponseHandlingException("Timeout"),
            None  # Success
        ]
        
        points = [Mock(spec=PointStruct)]
        uploader._upload_with_retry("test", points, batch_num=0)
        
        # Check sleep was called with increasing durations
        assert mock_sleep.call_count == 2
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert sleep_calls[0] == 1.0  # First retry: 1.0 * 1
        assert sleep_calls[1] == 2.0  # Second retry: 1.0 * 2
    
    def test_collection_name_passed_correctly(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test that collection name is passed to client."""
        uploader = DataUploader(mock_qdrant_client)
        
        uploader.upload_batch(
            collection_name="my_collection",
            dataset=sample_dataset,
            embeddings=sample_embeddings,
            show_progress=False
        )
        
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args[1]['collection_name'] == "my_collection"
    
    def test_payload_preserved(self, mock_qdrant_client, sample_dataset, sample_embeddings):
        """Test that dataset items are preserved as payload."""
        uploader = DataUploader(mock_qdrant_client)
        
        uploader.upload_batch(
            collection_name="test",
            dataset=sample_dataset,
            embeddings=sample_embeddings,
            show_progress=False
        )
        
        call_args = mock_qdrant_client.upsert.call_args
        points = call_args[1]['points']
        
        # Check first point payload matches first dataset item
        assert points[0].payload == sample_dataset[0]