from typing import List
import threading
import subprocess
import sys
import io
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def _execute_program(query: str) -> str:
    """
    Execute a single Python program and return its output.
    
    Args:
        query: Python program to execute as a string
    
    Returns:
        String containing both stdout and stderr outputs
    """
    result = ""
    
    try:
        # Create a separate process for execution using subprocess
        process = subprocess.Popen(
            [sys.executable, "-c", query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Capture output and errors
        stdout, stderr = process.communicate()
        
        # Combine the outputs
        result = stdout
        if stderr:
            result += f"\nERROR:\n{stderr}"
            
    except Exception as e:
        # Capture any exceptions that might occur during execution
        result = f"Error executing program: {str(e)}"
    
    return result


def batch_execute(queries: List[str] = None) -> List[str]:
    """
    Batchified programs to be executed using multithreading with progress tracking.
    
    Args:
        queries: Python programs to execute
        
    Returns:
        List of strings where each string contains the standard output 
        and standard error of the corresponding program
    """
    if queries is None or len(queries) == 0:
        return []
    
    results = [None] * len(queries)
    
    # Function to be executed by each thread
    def execute_and_store(index, query):
        results[index] = _execute_program(query)
    
    # Use ThreadPoolExecutor for thread management
    with ThreadPoolExecutor() as executor:
        # Create a list of futures
        futures = []
        for i, query in enumerate(queries):
            future = executor.submit(execute_and_store, i, query)
            futures.append(future)
        
        # Display progress bar
        for _ in tqdm(
            concurrent.futures.as_completed(futures), 
            total=len(futures),
            desc="Executing programs"
        ):
            pass
    
    return results