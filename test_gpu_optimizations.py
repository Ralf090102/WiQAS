#!/usr/bin/env python3
"""
Test GPU optimizations in WiQAS RAG pipeline.
Tests embedding generation, reranking, and overall performance improvements.
"""

import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Import WiQAS modules
from src.utilities.config import WiQASConfig
from src.utilities.gpu_utils import GPUManager, get_gpu_manager
from src.retrieval.embeddings import EmbeddingManager
from src.retrieval.reranker import RerankerManager, Document
from src.utilities.utils import log_info

console = Console()


def test_gpu_detection():
    """Test GPU detection and configuration"""
    console.print("\n[bold cyan]🔍 GPU Detection Test[/bold cyan]")
    
    config = WiQASConfig()
    gpu_manager = get_gpu_manager(config)
    
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("PyTorch CUDA Available", str(torch.cuda.is_available()))
    table.add_row("CUDA Version", str(torch.version.cuda) if torch.version.cuda else "N/A")
    table.add_row("cuDNN Available", str(torch.backends.cudnn.is_available()))
    table.add_row("Device Count", str(torch.cuda.device_count()))
    table.add_row("Selected Device", str(gpu_manager.get_device()))
    table.add_row("NVIDIA GPU Detected", str(gpu_manager.is_nvidia_gpu))
    
    if gpu_manager.is_nvidia_gpu:
        memory_info = gpu_manager.get_memory_info()
        if memory_info.get('memory_info'):
            mem = memory_info['memory_info']
            table.add_row("GPU Memory Total", f"{mem['total_mb']:.0f} MB")
            table.add_row("GPU Memory Free", f"{mem['total_mb'] - mem['allocated_mb']:.0f} MB")
    
    console.print(table)
    return gpu_manager


def test_embedding_optimizations():
    """Test GPU-optimized embedding generation"""
    console.print("\n[bold cyan]🚀 Embedding GPU Optimizations Test[/bold cyan]")
    
    config = WiQASConfig()
    embedding_manager = EmbeddingManager(config)
    
    # Test single embedding
    test_text = "Ang pagkakaisa ng bayan ay susi sa tagumpay ng demokratiya sa Pilipinas."
    
    start_time = time.time()
    embedding = embedding_manager.encode_single(test_text)
    single_time = time.time() - start_time
    
    # Test batch embeddings
    test_texts = [
        "Ang kultura ng Pilipinas ay mayaman at iba-iba.",
        "Bayanihan spirit is deeply rooted in Filipino communities.",
        "Education plays a crucial role in national development.",
        "Technology advancement requires skilled workforce.",
        "Sustainable development needs community participation."
    ]
    
    start_time = time.time()
    batch_embeddings = embedding_manager.encode_batch(test_texts)
    batch_time = time.time() - start_time
    
    # Display results
    model_info = embedding_manager.get_model_info()
    
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", model_info['model_name'])
    table.add_row("Device", model_info['device'])
    table.add_row("NVIDIA GPU", str(model_info['nvidia_gpu_detected']))
    table.add_row("Optimized Batch Size", str(model_info['optimized_batch_size']))
    table.add_row("Base Batch Size", str(model_info['base_batch_size']))
    table.add_row("Embedding Dimension", str(model_info['embedding_dimension']))
    table.add_row("Single Encoding Time", f"{single_time:.3f}s")
    table.add_row("Batch Encoding Time", f"{batch_time:.3f}s")
    table.add_row("Batch Throughput", f"{len(test_texts)/batch_time:.1f} texts/sec")
    
    console.print(table)
    
    if len(embedding) > 0:
        console.print(f"[green]✅ Single embedding generated: {len(embedding)} dimensions[/green]")
    else:
        console.print("[red]❌ Failed to generate single embedding[/red]")
        
    successful_embeddings = sum(1 for emb in batch_embeddings if len(emb) > 0)
    if successful_embeddings == len(test_texts):
        console.print(f"[green]✅ Batch embeddings generated: {successful_embeddings}/{len(test_texts)} successful[/green]")
    else:
        console.print(f"[yellow]⚠️ Partial batch success: {successful_embeddings}/{len(test_texts)} successful[/yellow]")
    
    return embedding_manager


def test_reranker_optimizations():
    """Test GPU-optimized reranking"""
    console.print("\n[bold cyan]🎯 Reranker GPU Optimizations Test[/bold cyan]")
    
    config = WiQASConfig()
    reranker = RerankerManager(config.rag.reranker)
    
    # Create test documents
    test_documents = [
        Document(
            content="Ang pagkakaisa at solidaridad ng mga Pilipino ay nagbubunga ng malakas na demokrasya.",
            metadata={"source": "test1.txt"},
            score=0.5,
            doc_id="doc1"
        ),
        Document(
            content="Filipino cultural values emphasize community cooperation and mutual support.",
            metadata={"source": "test2.txt"},
            score=0.6,
            doc_id="doc2"
        ),
        Document(
            content="Technology innovation drives economic growth in developing nations.",
            metadata={"source": "test3.txt"},
            score=0.4,
            doc_id="doc3"
        ),
        Document(
            content="Sustainable development requires balanced environmental and social policies.",
            metadata={"source": "test4.txt"},
            score=0.3,
            doc_id="doc4"
        )
    ]
    
    query = "What are Filipino cultural values and their impact on society?"
    
    start_time = time.time()
    reranked_docs = reranker.rerank_documents(query, test_documents, top_k=3)
    rerank_time = time.time() - start_time
    
    # Display results
    model_info = reranker.get_model_info()
    
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Model", model_info['model_name'])
    table.add_row("Device", model_info['device'])
    table.add_row("NVIDIA GPU", str(model_info['nvidia_gpu_detected']))
    table.add_row("Optimized Batch Size", str(model_info['optimized_batch_size']))
    table.add_row("Base Batch Size", str(model_info['base_batch_size']))
    table.add_row("Model Loaded", str(model_info['model_loaded']))
    table.add_row("Reranking Time", f"{rerank_time:.3f}s")
    table.add_row("Documents Processed", str(len(test_documents)))
    table.add_row("Top Results Returned", str(len(reranked_docs)))
    
    console.print(table)
    
    if reranked_docs:
        console.print(f"[green]✅ Reranking successful: {len(reranked_docs)} documents returned[/green]")
        
        # Show top results
        results_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
        results_table.add_column("Rank", style="yellow", width=6)
        results_table.add_column("Score", style="green", width=8)
        results_table.add_column("Content Preview", style="white")
        
        for i, doc in enumerate(reranked_docs, 1):
            preview = doc.content[:80] + "..." if len(doc.content) > 80 else doc.content
            results_table.add_row(str(i), f"{doc.score:.4f}", preview)
            
        console.print("\n[bold]Top Reranked Results:[/bold]")
        console.print(results_table)
    else:
        console.print("[red]❌ Reranking failed[/red]")
    
    return reranker


def test_end_to_end_performance():
    """Test complete pipeline performance"""
    console.print("\n[bold cyan]🏁 End-to-End Pipeline Performance Test[/bold cyan]")
    
    config = WiQASConfig()
    
    # Test components
    gpu_manager = get_gpu_manager(config)
    embedding_manager = EmbeddingManager(config)
    reranker = RerankerManager(config.rag.reranker)
    
    # Create a realistic test scenario
    query = "How does Filipino culture promote community solidarity?"
    documents = [
        "Ang bayanihan ay isang mahalagang tradisyon sa kultura ng mga Pilipino na nagpapakita ng pagkakaisa.",
        "Filipino communities practice cooperative values through various cultural traditions and festivals.", 
        "The concept of kapamilya extends beyond blood relations to include community members in Filipino society.",
        "Modern technology has changed how Filipino families communicate but core values remain strong.",
        "Economic development in the Philippines requires preserving cultural heritage while embracing progress.",
        "Environmental conservation efforts in Filipino communities often involve collective action and shared responsibility."
    ]
    
    start_time = time.time()
    
    # Step 1: Generate embeddings
    embeddings = embedding_manager.encode_batch(documents)
    embedding_time = time.time() - start_time
    
    # Step 2: Create documents for reranking
    docs_for_rerank = []
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        if len(emb) > 0:  # Only include successful embeddings
            docs_for_rerank.append(Document(
                content=doc,
                metadata={"source": f"doc_{i}.txt"},
                score=0.5 + i * 0.1,  # Simulate initial scores
                doc_id=f"doc_{i}"
            ))
    
    rerank_start = time.time()
    reranked_docs = reranker.rerank_documents(query, docs_for_rerank, top_k=3)
    rerank_time = time.time() - rerank_start
    
    total_time = time.time() - start_time
    
    # Display comprehensive results
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Pipeline Stage", style="cyan")
    table.add_column("Time", style="green")
    table.add_column("Throughput", style="yellow")
    table.add_column("Status", style="blue")
    
    successful_embeddings = sum(1 for emb in embeddings if len(emb) > 0)
    embedding_status = "✅ Success" if successful_embeddings == len(documents) else f"⚠️ {successful_embeddings}/{len(documents)}"
    rerank_status = "✅ Success" if reranked_docs else "❌ Failed"
    
    table.add_row(
        "Embedding Generation", 
        f"{embedding_time:.3f}s", 
        f"{len(documents)/embedding_time:.1f} docs/sec",
        embedding_status
    )
    table.add_row(
        "Document Reranking", 
        f"{rerank_time:.3f}s", 
        f"{len(docs_for_rerank)/rerank_time:.1f} docs/sec" if rerank_time > 0 else "N/A",
        rerank_status
    )
    table.add_row(
        "Total Pipeline", 
        f"{total_time:.3f}s", 
        f"{len(documents)/total_time:.1f} docs/sec",
        "✅ Complete"
    )
    
    console.print(table)
    
    # Device summary
    device_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE)
    device_table.add_column("Component", style="cyan")
    device_table.add_column("Device", style="green")
    device_table.add_column("GPU Accelerated", style="yellow")
    
    device_table.add_row("Embeddings", str(embedding_manager.device), str(embedding_manager.gpu_manager.is_nvidia_gpu))
    device_table.add_row("Reranking", reranker._device, str(reranker.gpu_manager.is_nvidia_gpu))
    device_table.add_row("Memory Management", str(gpu_manager.get_device()), str(gpu_manager.is_nvidia_gpu))
    
    console.print("\n[bold]Device Configuration:[/bold]")
    console.print(device_table)
    
    # Cleanup
    embedding_manager.cleanup()
    reranker.cleanup()
    
    return total_time


def main():
    """Run all GPU optimization tests"""
    console.print(Panel.fit(
        "[bold green]🚀 WiQAS GPU Acceleration Test Suite[/bold green]\n"
        "[cyan]Testing GPU optimizations in retrieval and generation pipeline[/cyan]",
        box=box.DOUBLE
    ))
    
    try:
        # Run all tests
        gpu_manager = test_gpu_detection()
        embedding_manager = test_embedding_optimizations()
        reranker = test_reranker_optimizations()
        total_time = test_end_to_end_performance()
        
        # Final summary
        console.print("\n" + "="*50)
        console.print("[bold green]🎉 GPU Optimization Test Complete![/bold green]")
        
        if gpu_manager.is_nvidia_gpu:
            console.print("[green]✅ NVIDIA GPU detected and optimized - WiQAS using GPU acceleration![/green]")
        else:
            console.print("[yellow]⚠️ No NVIDIA GPU detected - WiQAS using CPU with optimization fallbacks[/yellow]")
            console.print("[blue]💡 When GPU is available, WiQAS will automatically use it for acceleration[/blue]")
        
        console.print(f"[cyan]📊 Total pipeline time: {total_time:.3f}s[/cyan]")
        console.print("[dim]GPU optimizations include: Mixed precision, optimal batch sizing, memory management, and graceful CPU fallback[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Test failed: {e}[/red]")
        import traceback
        console.print(f"[red]{traceback.format_exc()}[/red]")


if __name__ == "__main__":
    main()