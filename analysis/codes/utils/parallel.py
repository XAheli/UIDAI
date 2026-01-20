"""
Parallel Processing Utilities
=============================
Author: Shuvam Banerji Seal's Team
Date: January 2026

High-performance multiprocessing utilities for data analysis.
Uses all available CPU cores for maximum throughput.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Union, Iterator, Tuple
import pandas as pd
import numpy as np
from functools import partial
from tqdm import tqdm
import logging

# Configure CPU count
CPU_COUNT = os.cpu_count() or 4
DEFAULT_CHUNK_SIZE = 50000

logger = logging.getLogger(__name__)


class ParallelProcessor:
    """
    High-performance parallel processor for dataframe operations.
    
    Supports both process-based and thread-based parallelism with
    automatic chunking and progress tracking.
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(
        self,
        n_workers: int = CPU_COUNT,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        use_threads: bool = False,
        show_progress: bool = True
    ):
        """
        Initialize parallel processor.
        
        Args:
            n_workers: Number of parallel workers (default: all CPUs)
            chunk_size: Records per chunk for processing
            use_threads: Use threads instead of processes (for I/O bound tasks)
            show_progress: Show tqdm progress bar
        """
        self.n_workers = n_workers
        self.chunk_size = chunk_size
        self.use_threads = use_threads
        self.show_progress = show_progress
        
        logger.info(f"ParallelProcessor initialized: {n_workers} workers, "
                   f"chunk_size={chunk_size}, threads={use_threads}")
    
    def _create_chunks(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Split dataframe into chunks for parallel processing."""
        n_chunks = max(1, len(df) // self.chunk_size + (1 if len(df) % self.chunk_size else 0))
        return np.array_split(df, n_chunks)
    
    def process_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        desc: str = "Processing"
    ) -> pd.DataFrame:
        """
        Process dataframe in parallel chunks.
        
        Args:
            df: Input dataframe
            func: Function to apply to each chunk
            desc: Progress bar description
            
        Returns:
            Processed dataframe
        """
        chunks = self._create_chunks(df)
        
        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        with ExecutorClass(max_workers=self.n_workers) as executor:
            if self.show_progress:
                results = list(tqdm(
                    executor.map(func, chunks),
                    total=len(chunks),
                    desc=desc
                ))
            else:
                results = list(executor.map(func, chunks))
        
        return pd.concat(results, ignore_index=True)
    
    def map_function(
        self,
        items: List[Any],
        func: Callable[[Any], Any],
        desc: str = "Mapping"
    ) -> List[Any]:
        """
        Apply function to list items in parallel.
        
        Args:
            items: List of items to process
            func: Function to apply to each item
            desc: Progress bar description
            
        Returns:
            List of results
        """
        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        with ExecutorClass(max_workers=self.n_workers) as executor:
            if self.show_progress:
                results = list(tqdm(
                    executor.map(func, items),
                    total=len(items),
                    desc=desc
                ))
            else:
                results = list(executor.map(func, items))
        
        return results
    
    def apply_with_callback(
        self,
        items: List[Any],
        func: Callable[[Any], Any],
        callback: Optional[Callable[[Any], None]] = None,
        desc: str = "Processing"
    ) -> List[Any]:
        """
        Apply function with callback for each completed item.
        
        Args:
            items: List of items to process
            func: Function to apply
            callback: Optional callback for each completed result
            desc: Progress bar description
            
        Returns:
            List of results in original order
        """
        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        results = [None] * len(items)
        
        with ExecutorClass(max_workers=self.n_workers) as executor:
            futures = {executor.submit(func, item): i for i, item in enumerate(items)}
            
            iterator = as_completed(futures)
            if self.show_progress:
                iterator = tqdm(iterator, total=len(items), desc=desc)
            
            for future in iterator:
                idx = futures[future]
                result = future.result()
                results[idx] = result
                
                if callback:
                    callback(result)
        
        return results


def parallel_apply(
    df: pd.DataFrame,
    func: Callable[[pd.DataFrame], pd.DataFrame],
    n_workers: int = CPU_COUNT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    desc: str = "Processing"
) -> pd.DataFrame:
    """
    Convenience function to apply a function to dataframe in parallel.
    
    Args:
        df: Input dataframe
        func: Function to apply
        n_workers: Number of workers
        chunk_size: Chunk size
        desc: Progress description
        
    Returns:
        Processed dataframe
    """
    processor = ParallelProcessor(
        n_workers=n_workers,
        chunk_size=chunk_size
    )
    return processor.process_dataframe(df, func, desc)


def parallel_map(
    items: List[Any],
    func: Callable[[Any], Any],
    n_workers: int = CPU_COUNT,
    use_threads: bool = False,
    desc: str = "Mapping"
) -> List[Any]:
    """
    Convenience function to map function over items in parallel.
    
    Args:
        items: List of items
        func: Function to apply
        n_workers: Number of workers
        use_threads: Use threads instead of processes
        desc: Progress description
        
    Returns:
        List of results
    """
    processor = ParallelProcessor(
        n_workers=n_workers,
        use_threads=use_threads
    )
    return processor.map_function(items, func, desc)


def parallel_groupby_apply(
    df: pd.DataFrame,
    groupby_cols: Union[str, List[str]],
    func: Callable[[pd.DataFrame], pd.DataFrame],
    n_workers: int = CPU_COUNT,
    desc: str = "Processing groups"
) -> pd.DataFrame:
    """
    Apply function to grouped dataframe in parallel.
    
    Args:
        df: Input dataframe
        groupby_cols: Column(s) to group by
        func: Function to apply to each group
        n_workers: Number of workers
        desc: Progress description
        
    Returns:
        Combined results
    """
    groups = [group for _, group in df.groupby(groupby_cols)]
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(tqdm(
            executor.map(func, groups),
            total=len(groups),
            desc=desc
        ))
    
    return pd.concat(results, ignore_index=True)


class ChunkedReader:
    """
    Memory-efficient chunked file reader with parallel processing.
    
    Author: Shuvam Banerji Seal's Team
    """
    
    def __init__(
        self,
        filepath: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        n_workers: int = CPU_COUNT
    ):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.n_workers = n_workers
    
    def process(
        self,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        desc: str = "Processing file"
    ) -> pd.DataFrame:
        """
        Process large file in chunks with parallel processing.
        
        Args:
            func: Function to apply to each chunk
            desc: Progress description
            
        Returns:
            Combined processed dataframe
        """
        # First pass: count total rows for progress
        total_rows = sum(1 for _ in open(self.filepath)) - 1
        n_chunks = total_rows // self.chunk_size + 1
        
        results = []
        chunks = pd.read_csv(self.filepath, chunksize=self.chunk_size)
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for chunk in chunks:
                futures.append(executor.submit(func, chunk))
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                results.append(future.result())
        
        return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    # Test parallel processing
    import time
    
    def slow_process(df):
        time.sleep(0.1)
        return df.copy()
    
    # Create test data
    test_df = pd.DataFrame({
        'a': range(100000),
        'b': range(100000)
    })
    
    print(f"Testing with {CPU_COUNT} workers...")
    
    start = time.time()
    result = parallel_apply(test_df, slow_process, desc="Test")
    elapsed = time.time() - start
    
    print(f"Processed {len(result)} rows in {elapsed:.2f}s")
