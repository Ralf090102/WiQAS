#!/usr/bin/env python3
"""
GPU Detection and Performance Test for WiQAS
Tests NVIDIA GPU detection, CUDA availability, and embedding performance.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import WiQAS modules
try:
    from src.utilities.gpu_utils import GPUManager, detect_gpu_info
    from src.utilities.config import WiQASConfig
    from src.retrieval.embeddings import EmbeddingManager

    WIQAS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import WiQAS modules: {e}")
    WIQAS_AVAILABLE = False

console = Console()


def test_basic_cuda():
    """Test basic CUDA availability"""
    console.print("\n[bold cyan]🔍 Basic CUDA Detection[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    # PyTorch CUDA info
    table.add_row("CUDA Available", str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        table.add_row("CUDA Version", str(torch.version.cuda))
        table.add_row("cuDNN Version", str(torch.backends.cudnn.version()))
        table.add_row("GPU Count", str(torch.cuda.device_count()))

        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            table.add_row(f"GPU {i} Name", gpu_name)

            # Memory info
            props = torch.cuda.get_device_properties(i)
            total_memory = props.total_memory / (1024**3)  # GB
            table.add_row(f"GPU {i} Memory", f"{total_memory:.1f} GB")

            # Current memory usage
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                reserved = torch.cuda.memory_reserved(i) / (1024**2)  # MB
                table.add_row(f"GPU {i} Allocated", f"{allocated:.1f} MB")
                table.add_row(f"GPU {i} Reserved", f"{reserved:.1f} MB")

    console.print(table)


def test_nvidia_ml():
    """Test nvidia-ml-py (if available)"""
    console.print("\n[bold cyan]🔍 NVIDIA ML Python Detection[/bold cyan]")

    try:
        import pynvml

        pynvml.nvmlInit()

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("GPU", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Memory", style="yellow")
        table.add_column("Temperature", style="red")
        table.add_column("Power", style="blue")

        gpu_count = pynvml.nvmlDeviceGetCount()

        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            # Handle both string and bytes return types
            if isinstance(name, bytes):
                name = name.decode()

            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = memory_info.used / (1024**2)  # MB
            memory_total = memory_info.total / (1024**2)  # MB
            memory_str = f"{memory_used:.0f}/{memory_total:.0f} MB"

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                temp_str = f"{temp}°C"
            except:
                temp_str = "N/A"

            # Power
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert mW to W
                power_str = f"{power:.1f}W"
            except:
                power_str = "N/A"

            table.add_row(f"GPU {i}", name, memory_str, temp_str, power_str)

        console.print(table)

    except ImportError:
        console.print("[yellow]⚠️  nvidia-ml-py not available. Install with: pip install nvidia-ml-py[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Error accessing NVIDIA ML: {e}[/red]")


def test_wiqas_gpu_detection():
    """Test WiQAS GPU detection"""
    if not WIQAS_AVAILABLE:
        console.print("\n[yellow]⚠️  WiQAS modules not available - skipping WiQAS GPU tests[/yellow]")
        return

    console.print("\n[bold cyan]🔍 WiQAS GPU Detection[/bold cyan]")

    try:
        # Test GPU manager
        config = WiQASConfig()
        gpu_manager = GPUManager(config)

        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("CUDA Available", str(gpu_manager.cuda_available))
        table.add_row("NVIDIA GPU Detected", str(gpu_manager.is_nvidia_gpu))
        table.add_row("Selected Device", str(gpu_manager.device))

        if gpu_manager.gpu_info:
            for gpu_id, info in gpu_manager.gpu_info.items():
                table.add_row(f"GPU {gpu_id} Name", info["name"])
                table.add_row(f"GPU {gpu_id} Memory", f"{info['memory_total'] / (1024**3):.1f} GB")

        # Test batch size optimization
        base_batch_size = 32
        optimal_batch_size = gpu_manager.get_optimal_batch_size(base_batch_size)
        table.add_row("Base Batch Size", str(base_batch_size))
        table.add_row("Optimal Batch Size", str(optimal_batch_size))

        console.print(table)

        # Test memory info
        memory_info = gpu_manager.get_memory_info()
        if memory_info["memory_info"]:
            mem = memory_info["memory_info"]
            console.print(f"\n[green]💾 GPU Memory: {mem['allocated_mb']:.0f}MB allocated, {mem['reserved_mb']:.0f}MB reserved, {mem['total_mb']:.0f}MB total[/green]")

    except Exception as e:
        console.print(f"[red]❌ Error testing WiQAS GPU detection: {e}[/red]")


def test_embedding_performance():
    """Test embedding performance with GPU vs CPU"""
    if not WIQAS_AVAILABLE:
        console.print("\n[yellow]⚠️  WiQAS modules not available - skipping embedding performance test[/yellow]")
        return

    console.print("\n[bold cyan]🚀 Embedding Performance Test[/bold cyan]")

    # Sample texts for testing
    test_texts = [
        "Ang kultura ng Pilipinas ay mayaman at iba-iba.",
        "Philippine culture is rich and diverse with influences from various civilizations.",
        "The Filipino people have a strong sense of community and family values.",
        "Ang mga Pilipino ay kilala sa kanilang pakikipagkapwa at malasakit sa iba.",
        "Traditional Filipino arts include weaving, pottery, and wood carving.",
    ] * 10  # 50 texts total

    try:
        config = WiQASConfig()
        embedding_manager = EmbeddingManager(config)

        console.print(f"[blue]🔧 Using device: {embedding_manager.device}[/blue]")
        console.print(f"[blue]🔧 Batch size: {embedding_manager.batch_size}[/blue]")

        # Test single encoding
        console.print("\n[yellow]Testing single text encoding...[/yellow]")
        start_time = time.time()
        embedding = embedding_manager.encode_single(test_texts[0])
        single_time = time.time() - start_time

        console.print(f"[green]✅ Single encoding: {single_time:.3f}s, dimension: {len(embedding)}[/green]")

        # Test batch encoding
        console.print(f"\n[yellow]Testing batch encoding ({len(test_texts)} texts)...[/yellow]")
        start_time = time.time()
        embeddings = embedding_manager.encode_batch(test_texts)
        batch_time = time.time() - start_time

        successful_embeddings = sum(1 for emb in embeddings if len(emb) > 0)
        console.print(f"[green]✅ Batch encoding: {batch_time:.3f}s, {successful_embeddings}/{len(test_texts)} successful[/green]")
        console.print(f"[green]📊 Throughput: {len(test_texts)/batch_time:.1f} texts/second[/green]")

        # Display model info
        model_info = embedding_manager.get_model_info()
        console.print(f"\n[blue]📋 Model: {model_info['model_name']}[/blue]")
        console.print(f"[blue]📋 Device: {model_info['device']}[/blue]")
        console.print(f"[blue]📋 NVIDIA GPU: {model_info['nvidia_gpu_detected']}[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Error testing embedding performance: {e}[/red]")
        import traceback

        console.print(f"[red]{traceback.format_exc()}[/red]")


def test_simple_gpu_operation():
    """Test simple GPU tensor operations"""
    console.print("\n[bold cyan]🧮 Simple GPU Operations Test[/bold cyan]")

    if not torch.cuda.is_available():
        console.print("[yellow]⚠️  CUDA not available - skipping GPU operations test[/yellow]")
        return

    try:
        # Create tensors on GPU
        device = torch.device("cuda")

        # Test tensor operations
        console.print("[yellow]Creating tensors on GPU...[/yellow]")
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)

        # Matrix multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        gpu_time = time.time() - start_time

        # Verify result
        result_sum = c.sum().item()

        console.print(f"[green]✅ GPU matrix multiplication: {gpu_time:.4f}s[/green]")
        console.print(f"[green]✅ Result verification: sum = {result_sum:.2f}[/green]")

        # Compare with CPU
        a_cpu = a.cpu()
        b_cpu = b.cpu()

        start_time = time.time()
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time

        speedup = cpu_time / gpu_time
        console.print(f"[blue]🚀 CPU time: {cpu_time:.4f}s[/blue]")
        console.print(f"[blue]🚀 GPU speedup: {speedup:.1f}x faster[/blue]")

    except Exception as e:
        console.print(f"[red]❌ Error testing GPU operations: {e}[/red]")


def main():
    """Run all GPU tests"""
    console.print(
        Panel.fit(
            "[bold green]🖥️  WiQAS GPU Detection and Performance Test[/bold green]\n" "[cyan]Testing NVIDIA GPU detection and acceleration capabilities[/cyan]",
            box=box.DOUBLE,
        )
    )

    # Run all tests
    test_basic_cuda()
    test_nvidia_ml()
    test_wiqas_gpu_detection()
    test_simple_gpu_operation()
    test_embedding_performance()

    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold green]🎉 GPU Test Complete![/bold green]")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "nvidia" in gpu_name.lower() or "geforce" in gpu_name.lower():
            console.print("[green]✅ NVIDIA GPU detected - WiQAS will use GPU acceleration![/green]")
        else:
            console.print("[yellow]⚠️  Non-NVIDIA GPU detected - WiQAS will use CPU fallback[/yellow]")
    else:
        console.print("[red]❌ No CUDA support - WiQAS will use CPU only[/red]")

    console.print("\n[dim]Run your WiQAS ingestion to see GPU acceleration in action![/dim]")


if __name__ == "__main__":
    main()
