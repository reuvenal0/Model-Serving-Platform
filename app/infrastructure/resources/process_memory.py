"""Current process RSS using Windows APIs or Linux procfs; no dependencies."""

import ctypes
import os
from pathlib import Path
import sys


def _windows_rss_bytes() -> int:
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage',
                'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage',
            )]

    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.argtypes = []
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise OSError('Process memory probe unavailable')
    return int(counters.WorkingSetSize)


def _linux_rss_bytes() -> int:
    # statm's second field is current resident pages, not peak RSS.
    fields = Path('/proc/self/statm').read_text(encoding='ascii').split()
    pages = int(fields[1])
    page_size = os.sysconf('SC_PAGE_SIZE')
    if pages < 0 or page_size <= 0:
        raise ValueError('Invalid process memory counters')
    return pages * page_size


def process_rss_bytes() -> int | None:
    """Return current RSS bytes; None explicitly means unsupported/unavailable.

    Linux requires mounted/readable procfs. Windows reports the current working
    set. This observes the whole process and is not a memory limit or exact
    attribution of shared/native allocations to individual models.
    """
    try:
        if sys.platform == 'win32':
            return _windows_rss_bytes()
        if sys.platform.startswith('linux'):
            return _linux_rss_bytes()
        return None
    except (OSError, ValueError, IndexError, AttributeError):
        return None
