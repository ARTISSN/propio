import time
import ctypes
from ctypes import wintypes

# Windows constants
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

def prevent_sleep():
    """Prevent Windows from going to sleep while training"""
    print("Preventing system sleep...")
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )

def allow_sleep():
    """Allow system to sleep again"""
    print("Restoring normal system sleep settings...")
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)

# Use in your training script:
if __name__ == "__main__":
    try:
        prevent_sleep()
        # Your training code here
    finally:
        allow_sleep()
