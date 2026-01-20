"""
Progress Tracking Utilities
===========================
Author: Shuvam Banerji Seal's Team
Date: January 2026

Progress tracking and timing utilities for long-running operations.
"""

import time
import logging
from contextlib import contextmanager
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime, timedelta
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Track progress of multi-step operations.
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.description = description
        self.current_step = 0
        self.start_time = None
        self.step_times = []
        self.pbar = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total_steps, desc=self.description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar:
            self.pbar.close()
        
        elapsed = time.time() - self.start_time
        logger.info(f"{self.description} completed in {elapsed:.2f}s")
        
        return False
    
    def step(self, description: Optional[str] = None):
        """Mark completion of a step."""
        self.current_step += 1
        step_time = time.time()
        
        if self.step_times:
            self.step_times.append(step_time - self.step_times[-1])
        else:
            self.step_times.append(step_time - self.start_time)
        
        if self.pbar:
            if description:
                self.pbar.set_description(description)
            self.pbar.update(1)
    
    def get_eta(self) -> Optional[timedelta]:
        """Get estimated time to completion."""
        if not self.step_times or self.current_step == 0:
            return None
        
        avg_step_time = sum(self.step_times) / len(self.step_times)
        remaining_steps = self.total_steps - self.current_step
        
        return timedelta(seconds=avg_step_time * remaining_steps)
    
    def get_summary(self) -> dict:
        """Get progress summary."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'completed_steps': self.current_step,
            'total_steps': self.total_steps,
            'percent_complete': (self.current_step / self.total_steps) * 100,
            'elapsed_seconds': elapsed,
            'eta': self.get_eta()
        }


@contextmanager
def timed_operation(name: str, log_level: int = logging.INFO):
    """
    Context manager for timing operations.
    
    Args:
        name: Operation name for logging
        log_level: Logging level
        
    Usage:
        with timed_operation("Data loading"):
            load_data()
    """
    start = time.time()
    logger.log(log_level, f"Starting: {name}")
    
    try:
        yield
    finally:
        elapsed = time.time() - start
        
        if elapsed < 60:
            time_str = f"{elapsed:.2f}s"
        elif elapsed < 3600:
            time_str = f"{elapsed/60:.2f}m"
        else:
            time_str = f"{elapsed/3600:.2f}h"
        
        logger.log(log_level, f"Completed: {name} ({time_str})")


def timed(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        
        return result
    
    return wrapper


class OperationTimer:
    """
    Timer for tracking multiple operations.
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(self):
        self.operations = {}
        self.running = {}
    
    def start(self, name: str):
        """Start timing an operation."""
        self.running[name] = time.time()
    
    def stop(self, name: str) -> float:
        """Stop timing and record the operation."""
        if name not in self.running:
            raise ValueError(f"Operation {name} was not started")
        
        elapsed = time.time() - self.running[name]
        del self.running[name]
        
        if name not in self.operations:
            self.operations[name] = []
        
        self.operations[name].append(elapsed)
        
        return elapsed
    
    def get_stats(self, name: str) -> dict:
        """Get statistics for an operation."""
        if name not in self.operations:
            return None
        
        times = self.operations[name]
        
        return {
            'count': len(times),
            'total': sum(times),
            'mean': sum(times) / len(times),
            'min': min(times),
            'max': max(times)
        }
    
    def get_all_stats(self) -> dict:
        """Get statistics for all operations."""
        return {name: self.get_stats(name) for name in self.operations}
    
    def report(self) -> str:
        """Generate a timing report."""
        lines = ["=" * 50, "Operation Timing Report", "=" * 50]
        
        for name, times in self.operations.items():
            total = sum(times)
            mean = total / len(times)
            lines.append(f"{name}:")
            lines.append(f"  Count: {len(times)}")
            lines.append(f"  Total: {total:.2f}s")
            lines.append(f"  Mean: {mean:.2f}s")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


class ProgressCallback:
    """
    Callback for progress updates.
    
    Useful for integration with web frontends.
    """
    
    def __init__(self, callback: Optional[Callable[[dict], None]] = None):
        self.callback = callback
        self.total = 0
        self.current = 0
        self.message = ""
    
    def set_total(self, total: int):
        """Set total number of items."""
        self.total = total
        self._notify()
    
    def update(self, count: int = 1, message: str = ""):
        """Update progress."""
        self.current += count
        self.message = message
        self._notify()
    
    def _notify(self):
        """Send progress notification."""
        if self.callback:
            self.callback({
                'current': self.current,
                'total': self.total,
                'percent': (self.current / self.total * 100) if self.total > 0 else 0,
                'message': self.message
            })


if __name__ == "__main__":
    # Test progress tracking
    with ProgressTracker(10, "Test Processing") as tracker:
        for i in range(10):
            time.sleep(0.1)
            tracker.step(f"Step {i+1}")
    
    # Test timed operation
    with timed_operation("Test Operation"):
        time.sleep(0.5)
    
    # Test operation timer
    timer = OperationTimer()
    for i in range(5):
        timer.start("test_op")
        time.sleep(0.1)
        timer.stop("test_op")
    
    print(timer.report())
