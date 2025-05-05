"""Utility helpers for InsightForge."""
import os
import shutil

def ensure_dir(path: str):
    """Make `path` if it does not exist."""
    os.makedirs(path, exist_ok=True)

def move_file(src: str, dst_dir: str) -> str:
    """Move file `src` into directory `dst_dir`, returning the new path."""
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.move(src, dst)
    return dst

def is_code_error(output: str) -> bool:
    """Detect if a captured REPL output contains a Python error."""
    output = output.lower()
    return any(
        keyword in output
        for keyword in ["traceback", "error", "exception"]
    )
