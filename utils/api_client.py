"""
OVIP - API Client Utility
A robust HTTP client wrapper with automatic retry and backoff logic.
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)

class OVIPAPIClient:
    def __init__(self, base_url: str = "", retries: int = 3, backoff_factor: float = 1.0):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Implement robust retry strategy for transient network errors
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def get(self, endpoint: str, **kwargs):
        """Standard GET request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API GET Error ({url}): {e}")
            return None

    def post(self, endpoint: str, **kwargs):
        """Standard POST request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"API POST Error ({url}): {e}")
            return None
