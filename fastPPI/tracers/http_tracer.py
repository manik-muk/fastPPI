"""
HTTP request tracer for FastPPI.
Tracks requests.get() and response.json() calls for HTTP data loading patterns.
"""

from typing import Dict, Optional, Callable

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


class HTTPTracer:
    """Tracks HTTP requests and responses for compilation."""
    
    def __init__(self):
        self.enabled = False
        # Track pending HTTP requests: response_id -> url
        self.pending_http_requests: Dict[int, str] = {}
        # Track JSON responses: json_data_id -> response_id
        self.json_responses: Dict[int, int] = {}
        # Store original functions for unpatching
        self.original_functions: Dict[str, Callable] = {}
    
    def start_tracing(self):
        """Enable HTTP tracing and patch requests module."""
        if not REQUESTS_AVAILABLE:
            return
        
        self.enabled = True
        self._patch_requests()
    
    def stop_tracing(self):
        """Disable HTTP tracing and restore original functions."""
        if not REQUESTS_AVAILABLE:
            return
        
        self.enabled = False
        self._unpatch_requests()
        # Clear tracking data
        self.pending_http_requests = {}
        self.json_responses = {}
    
    def _patch_requests(self):
        """Patch requests.get and response.json() to track HTTP calls."""
        if not REQUESTS_AVAILABLE:
            return
        
        # Store original functions
        self.original_functions['requests_get'] = requests.get
        requests.get = self._wrap_requests_get('requests_get', self.original_functions['requests_get'])
        
        # Also patch response.json() method
        if hasattr(requests.Response, 'json'):
            self.original_functions['response_json'] = requests.Response.json
            requests.Response.json = self._wrap_response_json('response_json', self.original_functions['response_json'])
    
    def _unpatch_requests(self):
        """Restore original requests functions."""
        if not REQUESTS_AVAILABLE:
            return
        
        if 'requests_get' in self.original_functions:
            requests.get = self.original_functions['requests_get']
        if 'response_json' in self.original_functions:
            requests.Response.json = self.original_functions['response_json']
    
    def _wrap_requests_get(self, name: str, original_func: Callable) -> Callable:
        """Wrap requests.get to track HTTP requests."""
        def wrapped(*args, **kwargs):
            if not self.enabled:
                return original_func(*args, **kwargs)
            
            # Execute the original function
            response = original_func(*args, **kwargs)
            
            # Store the URL for this response
            if len(args) > 0 and isinstance(args[0], str):
                url = args[0]
                self.pending_http_requests[id(response)] = url
            
            return response
        return wrapped
    
    def _wrap_response_json(self, name: str, original_func: Callable) -> Callable:
        """Wrap response.json() to track JSON data from HTTP responses."""
        def wrapped(self_obj, *args, **kwargs):
            if not self.enabled:
                return original_func(self_obj, *args, **kwargs)
            
            # Execute the original function
            json_data = original_func(self_obj, *args, **kwargs)
            
            # Track which response produced this JSON data
            response_id = id(self_obj)
            json_id = id(json_data)
            self.json_responses[json_id] = response_id
            
            return json_data
        return wrapped
    
    def get_url_for_json(self, json_data) -> Optional[str]:
        """Get the URL that produced the given JSON data, if tracked."""
        json_id = id(json_data)
        if json_id in self.json_responses:
            response_id = self.json_responses[json_id]
            if response_id in self.pending_http_requests:
                return self.pending_http_requests[response_id]
        return None

