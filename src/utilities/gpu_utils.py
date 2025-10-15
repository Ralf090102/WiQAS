"""
GPU detection and device management utilities for WiQAS.

Provides NVIDIA GPU detection, device selection, and optimization settings
for accelerated document ingestion and embedding generation.
"""

import subprocess

import torch
import torch.backends.cudnn as cudnn
from torch import amp

from src.utilities.utils import log_debug, log_error, log_info, log_warning


class GPUManager:
    """
    Manages GPU detection, device selection, and optimization for WiQAS.

    Automatically detects NVIDIA GPUs and falls back to CPU for AMD GPUs
    or systems without CUDA support.
    """

    def __init__(self, config=None):
        """
        Initialize GPU manager.

        Args:
            config: WiQAS configuration object
        """
        self.config = config
        self.device = None
        self.gpu_info = None
        self.is_nvidia_gpu = False
        self.cuda_available = False

        self._detect_gpu()
        self._setup_optimizations()

    def _detect_gpu(self) -> None:
        """Detect available GPU and determine if it's NVIDIA."""
        try:
            # Check if CUDA is available
            self.cuda_available = torch.cuda.is_available()

            if not self.cuda_available:
                log_info("CUDA not available, using CPU", config=self.config)
                self.device = torch.device("cpu")
                return

            # Get GPU information
            self.gpu_info = self._get_gpu_info()

            # Check if it's an NVIDIA GPU
            self.is_nvidia_gpu = self._is_nvidia_gpu()

            if self.is_nvidia_gpu:
                # Use NVIDIA GPU
                gpu_count = torch.cuda.device_count()
                device_id = 0  # Use first GPU by default

                self.device = torch.device(f"cuda:{device_id}")
                gpu_name = torch.cuda.get_device_name(device_id)

                log_info(
                    f"Using NVIDIA GPU: {gpu_name} " f"({gpu_count} GPU{'s' if gpu_count > 1 else ''} available)",
                    config=self.config,
                )

                # Log GPU memory info
                if self.config and hasattr(self.config, "logging") and self.config.logging.verbose:
                    self._log_gpu_memory_info(device_id)

            else:
                # AMD GPU or unsupported GPU detected, fall back to CPU
                log_warning("Non-NVIDIA GPU detected (likely AMD). Falling back to CPU for compatibility.", config=self.config)
                self.device = torch.device("cpu")

        except Exception as e:
            log_error(f"GPU detection failed: {e}", config=self.config)
            self.device = torch.device("cpu")

    def _get_gpu_info(self) -> dict | None:
        """Get detailed GPU information using nvidia-ml-py if available."""
        try:
            import pynvml

            pynvml.nvmlInit()

            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_info = {}

            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                gpu_info[i] = {
                    "name": name,
                    "memory_total": memory_info.total,
                    "memory_free": memory_info.free,
                    "memory_used": memory_info.used,
                }

            return gpu_info

        except ImportError:
            # pynvml not available, use nvidia-smi fallback
            return self._get_gpu_info_nvidia_smi()
        except Exception as e:
            log_debug(f"Failed to get detailed GPU info: {e}", config=self.config)
            return None

    def _get_gpu_info_nvidia_smi(self) -> dict | None:
        """Get GPU information using nvidia-smi command."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                gpu_info = {}
                lines = result.stdout.strip().split("\n")

                for i, line in enumerate(lines):
                    parts = [part.strip() for part in line.split(",")]
                    if len(parts) >= 4:
                        gpu_info[i] = {
                            "name": parts[0],
                            "memory_total": int(parts[1]) * 1024 * 1024,  # Convert MB to bytes
                            "memory_free": int(parts[2]) * 1024 * 1024,
                            "memory_used": int(parts[3]) * 1024 * 1024,
                        }

                return gpu_info

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass
        except Exception as e:
            log_debug(f"nvidia-smi query failed: {e}", config=self.config)

        return None

    def _is_nvidia_gpu(self) -> bool:
        """Determine if the available GPU is NVIDIA."""
        try:
            # Try to get NVIDIA GPU name
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0).upper()

                # Check for NVIDIA in the name
                if "NVIDIA" in gpu_name or "GEFORCE" in gpu_name or "QUADRO" in gpu_name or "TESLA" in gpu_name:
                    return True

                # Check for AMD indicators
                if "AMD" in gpu_name or "RADEON" in gpu_name:
                    log_info(f"AMD GPU detected: {gpu_name}", config=self.config)
                    return False

                # If uncertain, try a small CUDA operation
                try:
                    test_tensor = torch.randn(10, 10, device="cuda")
                    _ = test_tensor.sum()
                    return True
                except Exception:
                    return False

        except Exception as e:
            log_debug(f"NVIDIA GPU detection failed: {e}", config=self.config)

        return False

    def _setup_optimizations(self) -> None:
        """Setup GPU optimizations for NVIDIA GPUs."""
        if not self.is_nvidia_gpu:
            return

        try:
            # Enable cuDNN benchmark for consistent input sizes
            if torch.backends.cudnn.is_available():
                cudnn.benchmark = True
                cudnn.deterministic = False  # Allow non-deterministic algorithms for speed
                log_debug("Enabled cuDNN optimizations", config=self.config)

            # Enable TensorFloat-32 (TF32) on Ampere GPUs for faster training
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                log_debug("Enabled TF32 optimizations", config=self.config)

            # Set memory fraction to avoid OOM errors (use config value if provided)
            if torch.cuda.is_available():
                try:
                    # Determine device id to apply memory fraction to
                    device_id = 0
                    if isinstance(self.device, torch.device) and self.device.type == "cuda":
                        try:
                            device_id = self.device.index if self.device.index is not None else torch.cuda.current_device()
                        except Exception:
                            device_id = torch.cuda.current_device()

                    # Get memory fraction from config or use default 0.9
                    fraction = 0.9
                    if self.config and hasattr(self.config, "gpu"):
                        fraction = float(getattr(self.config.gpu, "memory_fraction", fraction))

                    # Clamp fraction to sensible range
                    fraction = max(0.1, min(1.0, fraction))

                    torch.cuda.set_per_process_memory_fraction(fraction, device=device_id)
                    log_debug(f"Set per-process GPU memory fraction to {fraction}", config=self.config)
                except Exception as e:
                    log_debug(f"Failed to set per-process memory fraction: {e}", config=self.config)

        except Exception as e:
            log_warning(f"Failed to setup GPU optimizations: {e}", config=self.config)

    def _log_gpu_memory_info(self, device_id: int = 0) -> None:
        """Log GPU memory information."""
        try:
            if torch.cuda.is_available():
                # Get memory info in MB
                total_memory = torch.cuda.get_device_properties(device_id).total_memory / (1024**2)
                allocated_memory = torch.cuda.memory_allocated(device_id) / (1024**2)
                cached_memory = torch.cuda.memory_reserved(device_id) / (1024**2)

                log_debug(
                    f"GPU Memory - Total: {total_memory:.0f}MB, " f"Allocated: {allocated_memory:.0f}MB, " f"Cached: {cached_memory:.0f}MB",
                    config=self.config,
                )

        except Exception as e:
            log_debug(f"Failed to log GPU memory info: {e}", config=self.config)

    def get_device(self) -> torch.device:
        """Get the selected device (CPU or CUDA)."""
        return self.device

    def get_optimal_batch_size(self, base_batch_size: int) -> int:
        """
        Get optimal batch size based on available GPU memory.

        Args:
            base_batch_size: Base batch size for CPU

        Returns:
            Optimized batch size for current device
        """
        if not self.is_nvidia_gpu:
            return base_batch_size

        try:
            # Get available GPU memory in GB
            if self.gpu_info and 0 in self.gpu_info:
                available_memory_gb = self.gpu_info[0]["memory_free"] / (1024**3)
            else:
                # Fallback to PyTorch memory info
                device_props = torch.cuda.get_device_properties(0)
                total_memory_gb = device_props.total_memory / (1024**3)
                available_memory_gb = total_memory_gb * 0.8  # Conservative estimate

            # Scale batch size based on available memory
            # Base assumption: 8GB can handle 2x base batch size
            memory_factor = max(1, available_memory_gb / 4.0)
            optimal_batch_size = int(base_batch_size * memory_factor)

            # Cap at reasonable maximum to avoid OOM
            max_batch_size = base_batch_size * 4
            optimal_batch_size = min(optimal_batch_size, max_batch_size)

            log_debug(
                f"Optimized batch size: {optimal_batch_size} " f"(base: {base_batch_size}, memory factor: {memory_factor:.1f})",
                config=self.config,
            )

            return optimal_batch_size

        except Exception as e:
            log_warning(f"Failed to optimize batch size: {e}", config=self.config)
            return base_batch_size

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        if self.is_nvidia_gpu and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                log_debug("Cleared GPU memory cache", config=self.config)
            except Exception as e:
                log_warning(f"Failed to clear GPU cache: {e}", config=self.config)

    def get_memory_info(self) -> dict:
        """Get current GPU memory usage information."""
        if not self.is_nvidia_gpu or not torch.cuda.is_available():
            return {"device": "cpu", "memory_info": None}

        try:
            device_id = self.device.index if self.device.index is not None else 0

            return {
                "device": str(self.device),
                "memory_info": {
                    "allocated_mb": torch.cuda.memory_allocated(device_id) / (1024**2),
                    "reserved_mb": torch.cuda.memory_reserved(device_id) / (1024**2),
                    "total_mb": torch.cuda.get_device_properties(device_id).total_memory / (1024**2),
                },
            }

        except Exception as e:
            log_warning(f"Failed to get memory info: {e}", config=self.config)
            return {"device": str(self.device), "memory_info": None}

    def enable_mixed_precision(self):
        """Return context manager for mixed precision (torch.amp.autocast) on NVIDIA GPUs.

        Usage:
            with gpu_manager.enable_mixed_precision():
                # run inference
        """
        try:
            # Respect config flag if provided, otherwise default to True
            config_enable = True
            if self.config and hasattr(self.config, "gpu"):
                config_enable = getattr(self.config.gpu, "enable_mixed_precision", True)

            enabled = bool(self.is_nvidia_gpu and config_enable)
            # amp.autocast requires device_type; use 'cuda' when GPU is available
            return amp.autocast(device_type="cuda", enabled=enabled)
        except Exception:
            # Fallback to a no-op context manager if anything goes wrong
            class NoOpContextManager:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return NoOpContextManager()


def detect_gpu_info() -> tuple[bool, str | None, dict | None]:
    """
    Quick GPU detection function.

    Returns:
        Tuple of (is_nvidia_gpu, device_name, memory_info)
    """
    try:
        if not torch.cuda.is_available():
            return False, "CPU", None

        device_name = torch.cuda.get_device_name(0)

        # Check if NVIDIA
        is_nvidia = any(keyword in device_name.upper() for keyword in ["NVIDIA", "GEFORCE", "QUADRO", "TESLA"])

        memory_info = None
        if is_nvidia:
            try:
                props = torch.cuda.get_device_properties(0)
                memory_info = {
                    "total_mb": props.total_memory / (1024**2),
                    "allocated_mb": torch.cuda.memory_allocated(0) / (1024**2),
                }
            except Exception:
                pass

        return is_nvidia, device_name, memory_info

    except Exception:
        return False, "CPU", None


# Global GPU manager instance
_gpu_manager = None


def get_gpu_manager(config=None) -> GPUManager:
    """Get or create global GPU manager instance."""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager(config)
    return _gpu_manager
