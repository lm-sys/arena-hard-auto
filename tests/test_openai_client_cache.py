"""Unit tests for OpenAI client caching functionality."""

import unittest
from unittest import mock


class TestOpenAIClientCache(unittest.TestCase):
    """Test suite for the OpenAI client caching mechanism."""

    def setUp(self):
        """Reset thread-local cache before each test to ensure isolation."""
        from utils import openai_client
        # Ensure no cached client from earlier tests in this thread.
        openai_client._thread_local.__dict__.pop("openai_clients", None)

    def test_openai_client_is_cached_per_thread_and_endpoint(self):
        """Verify that OpenAI clients are cached per thread and endpoint configuration.

        Tests that:
        - Repeated calls with identical config return the same client instance
        - Different configurations (e.g., timeout) create separate cached clients
        """
        # Avoid importing the full completion stack (pulls optional deps).
        # Instead, mock an `openai` module and test the small helper directly.
        from utils import openai_client

        fake_openai = mock.Mock()
        fake_openai.OpenAI.side_effect = [object(), object(), object()]

        with mock.patch.dict("sys.modules", {"openai": fake_openai}):
            c1 = openai_client.get_openai_client(
                {"api_base": "http://localhost:8000/v1", "api_key": "k", "timeout": 123}
            )
            c2 = openai_client.get_openai_client(
                {"api_base": "http://localhost:8000/v1", "api_key": "k", "timeout": 123}
            )
            self.assertIs(c1, c2)
            self.assertEqual(fake_openai.OpenAI.call_count, 1)

            # Different timeout should create a new client
            c3 = openai_client.get_openai_client(
                {"api_base": "http://localhost:8000/v1", "api_key": "k", "timeout": 456}
            )
            self.assertIsNot(c1, c3)
            self.assertEqual(fake_openai.OpenAI.call_count, 2)


if __name__ == "__main__":
    unittest.main()

