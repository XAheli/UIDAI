"""
Base API Client Module
Author: Shuvam Banerji Seal's Team
UIDAI Hackathon - Aadhaar Data Analysis

Provides base classes and utilities for API clients:
- BaseAPIClient: Abstract base class for API clients
- RateLimiter: Rate limiting implementation
- APICache: Caching layer for API responses
"""

import time
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque
import threading

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import requests
try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests library not available")


class RateLimiter:
    """
    Token bucket rate limiter for API requests
    """
    
    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: int = 20
    ):
        """
        Initialize rate limiter
        
        Args:
            requests_per_second: Maximum sustained request rate
            burst_size: Maximum burst size
        """
        self.rate = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """
        Acquire a token, blocking if necessary
        
        Returns:
            True when token is acquired
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            # Wait for token
            wait_time = (1 - self.tokens) / self.rate
            time.sleep(wait_time)
            self.tokens = 0
            self.last_update = time.time()
            return True
    
    def try_acquire(self) -> bool:
        """
        Try to acquire a token without blocking
        
        Returns:
            True if token acquired, False otherwise
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            
            self.tokens = min(
                self.burst_size,
                self.tokens + elapsed * self.rate
            )
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            
            return False


class APICache:
    """
    Simple file-based cache for API responses
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = '.api_cache',
        default_ttl: int = 3600,  # 1 hour
        max_size: int = 10000
    ):
        """
        Initialize API cache
        
        Args:
            cache_dir: Directory for cache files
            default_ttl: Default time-to-live in seconds
            max_size: Maximum number of cached items
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_ttl = default_ttl
        self.max_size = max_size
        
        # In-memory cache
        self.memory_cache: Dict[str, Dict] = {}
        self.access_order = deque()
        self.lock = threading.Lock()
        
        self._load_disk_cache()
    
    def _generate_key(self, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from URL and params"""
        key_data = {'url': url, 'params': params or {}}
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_disk_cache(self):
        """Load existing cache from disk"""
        index_file = self.cache_dir / 'index.json'
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                # Load valid entries
                now = datetime.now().timestamp()
                for key, meta in index.items():
                    if meta['expires'] > now:
                        cache_file = self.cache_dir / f'{key}.json'
                        if cache_file.exists():
                            with open(cache_file, 'r') as f:
                                self.memory_cache[key] = {
                                    'data': json.load(f),
                                    'expires': meta['expires']
                                }
                                self.access_order.append(key)
                
                logger.info(f"Loaded {len(self.memory_cache)} items from disk cache")
            except Exception as e:
                logger.warning(f"Error loading disk cache: {str(e)}")
    
    def _save_disk_cache(self):
        """Save cache index to disk"""
        index = {
            key: {'expires': value['expires']}
            for key, value in self.memory_cache.items()
        }
        
        index_file = self.cache_dir / 'index.json'
        with open(index_file, 'w') as f:
            json.dump(index, f)
    
    def _evict_if_needed(self):
        """Evict oldest items if cache is full"""
        while len(self.memory_cache) >= self.max_size and self.access_order:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]
                
                # Remove from disk
                cache_file = self.cache_dir / f'{oldest_key}.json'
                if cache_file.exists():
                    cache_file.unlink()
    
    def get(
        self,
        url: str,
        params: Optional[Dict] = None
    ) -> Optional[Any]:
        """
        Get item from cache
        
        Args:
            url: Request URL
            params: Request parameters
            
        Returns:
            Cached data or None if not found/expired
        """
        key = self._generate_key(url, params)
        
        with self.lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check expiration
                if entry['expires'] > datetime.now().timestamp():
                    return entry['data']
                
                # Expired, remove
                del self.memory_cache[key]
        
        return None
    
    def set(
        self,
        url: str,
        data: Any,
        params: Optional[Dict] = None,
        ttl: Optional[int] = None
    ):
        """
        Set item in cache
        
        Args:
            url: Request URL
            data: Data to cache
            params: Request parameters
            ttl: Time-to-live in seconds
        """
        key = self._generate_key(url, params)
        ttl = ttl or self.default_ttl
        expires = datetime.now().timestamp() + ttl
        
        with self.lock:
            self._evict_if_needed()
            
            self.memory_cache[key] = {
                'data': data,
                'expires': expires
            }
            self.access_order.append(key)
            
            # Save to disk
            cache_file = self.cache_dir / f'{key}.json'
            with open(cache_file, 'w') as f:
                json.dump(data, f)
            
            self._save_disk_cache()
    
    def clear(self):
        """Clear all cached items"""
        with self.lock:
            self.memory_cache.clear()
            self.access_order.clear()
            
            # Clear disk cache
            for cache_file in self.cache_dir.glob('*.json'):
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'items': len(self.memory_cache),
            'max_size': self.max_size,
            'cache_dir': str(self.cache_dir)
        }


class BaseAPIClient(ABC):
    """
    Abstract base class for API clients
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        rate_limit: float = 10.0,
        cache_ttl: int = 3600,
        timeout: int = 30,
        retries: int = 3
    ):
        """
        Initialize API client
        
        Args:
            base_url: Base URL for API
            api_key: API key (if required)
            rate_limit: Requests per second
            cache_ttl: Cache time-to-live in seconds
            timeout: Request timeout in seconds
            retries: Number of retries for failed requests
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required")
        
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(rate_limit)
        
        # Set up cache
        cache_dir = Path('.api_cache') / self.__class__.__name__.lower()
        self.cache = APICache(cache_dir=cache_dir, default_ttl=cache_ttl)
        
        # Set up session with retries
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        # Statistics
        self.stats = {
            'requests': 0,
            'cache_hits': 0,
            'errors': 0
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers"""
        headers = {
            'User-Agent': 'UIDAI-Analysis/1.0 (Shuvam Banerji Seal Team)',
            'Accept': 'application/json'
        }
        
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def _make_request(
        self,
        endpoint: str,
        method: str = 'GET',
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Make API request with rate limiting and caching
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            params: Query parameters
            data: Request body
            use_cache: Whether to use cache
            
        Returns:
            Response data
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        # Check cache for GET requests
        if method == 'GET' and use_cache:
            cached = self.cache.get(url, params)
            if cached is not None:
                self.stats['cache_hits'] += 1
                return cached
        
        # Rate limit
        self.rate_limiter.acquire()
        
        try:
            self.stats['requests'] += 1
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=self._get_headers(),
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Cache successful GET requests
            if method == 'GET' and use_cache:
                self.cache.set(url, result, params)
            
            return result
            
        except requests.exceptions.RequestException as e:
            self.stats['errors'] += 1
            logger.error(f"API request failed: {str(e)}")
            raise
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if API is available"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            **self.stats,
            'cache': self.cache.get_stats(),
            'rate_limit': self.rate_limiter.rate
        }


# Export classes
__all__ = ['BaseAPIClient', 'RateLimiter', 'APICache']
