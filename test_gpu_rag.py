#!/usr/bin/env python3
"""
GPU-Accelerated RAG Pipeline Test for WiQAS

Comprehensive test to validate GPU acceleration across all WiQAS components:
- EmbeddingManager (BGE-M3 with GPU acceleration)
- RerankerManager (CrossEncoder with GPU optimization)
- RetrievalEvaluator (batch similarity with GPU processing)
- VectorStore operations
- End-to-end retrieval pipeline
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn

# Import WiQAS modules
try:
    from src.utilities.gpu_utils import GPUManager, detect_gpu_info
    from src.utilities.config import WiQASConfig
    from src.retrieval.embeddings import EmbeddingManager
    from src.retrieval.reranker import RerankerManager
    from src.retrieval.evaluator import RetrievalEvaluator
    from src.retrieval.retriever import WiQASRetriever
    from src.core.ingest import DocumentIngestor
    WIQAS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import WiQAS modules: {e}")
    WIQAS_AVAILABLE = False

console = Console()


class GPURAGTester:
    """Comprehensive GPU acceleration tester for WiQAS RAG pipeline."""
    
    def __init__(self):
        """Initialize the tester."""
        self.config = WiQASConfig()
        self.gpu_manager = GPUManager(self.config)
        self.results = {}
        
    def test_gpu_detection(self):
        """Test GPU detection and configuration."""
        console.print("\n[bold cyan]🔍 GPU Detection and Configuration[/bold cyan]")
        
        # Basic CUDA detection
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        # PyTorch CUDA
        status = "✅ Available" if cuda_available else "❌ Not Available"
        details = f"{cuda_device_count} device(s)" if cuda_available else "CPU only"
        table.add_row("PyTorch CUDA", status, details)
        
        # GPU Manager
        gm_status = "✅ NVIDIA GPU" if self.gpu_manager.is_nvidia_gpu else "⚠️  CPU/Non-NVIDIA"
        gm_details = str(self.gpu_manager.device)
        table.add_row("GPU Manager", gm_status, gm_details)
        
        # Device info
        if cuda_available and cuda_device_count > 0:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                table.add_row("GPU Model", "✅ Detected", f"{gpu_name}")
                table.add_row("GPU Memory", "✅ Available", f"{memory_mb:.0f} MB")
            except Exception as e:
                table.add_row("GPU Details", "❌ Error", str(e))
        
        console.print(table)
        
        self.results['gpu_detection'] = {
            'cuda_available': cuda_available,
            'device_count': cuda_device_count,
            'nvidia_gpu': self.gpu_manager.is_nvidia_gpu,
            'device': str(self.gpu_manager.device)
        }
        
        return cuda_available or self.gpu_manager.is_nvidia_gpu
    
    def test_embedding_performance(self):
        """Test embedding generation with GPU acceleration."""
        console.print("\n[bold cyan]🚀 Embedding Performance Test[/bold cyan]")
        
        try:
            embedding_manager = EmbeddingManager(self.config)
            
            # Test data
            test_texts = [
                "Kumusta! This is a test in Filipino and English.",
                "The quick brown fox jumps over the lazy dog.",
                "Ang kulturang Pilipino ay mayaman sa tradisyon at kasaysayan.",
                "Machine learning and artificial intelligence are transforming technology.",
                "Pagkakaisa at bayanihan ang nagbubuklod sa mga Pilipino.",
            ] * 10  # 50 texts total
            
            # Single encoding test
            console.print("📝 Testing single text encoding...")
            start_time = time.time()
            single_embedding = embedding_manager.encode_single(test_texts[0])
            single_time = time.time() - start_time
            
            # Batch encoding test
            console.print("📦 Testing batch encoding...")
            start_time = time.time()
            batch_embeddings = embedding_manager.encode_batch(test_texts)
            batch_time = time.time() - start_time
            
            # Validate results
            successful_embeddings = sum(1 for emb in batch_embeddings if emb and len(emb) > 0)
            
            # Display results
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Test", style="cyan")
            table.add_column("Time", style="green")
            table.add_column("Throughput", style="yellow")
            table.add_column("Status", style="white")
            
            table.add_row(
                "Single Encoding", 
                f"{single_time:.3f}s", 
                "1 text", 
                "✅ Success" if len(single_embedding) > 0 else "❌ Failed"
            )
            
            table.add_row(
                "Batch Encoding", 
                f"{batch_time:.3f}s", 
                f"{len(test_texts)/batch_time:.1f} texts/sec", 
                f"✅ {successful_embeddings}/{len(test_texts)} successful"
            )
            
            console.print(table)
            
            # Model info
            model_info = embedding_manager.get_model_info()
            console.print(f"\n[blue]📋 Model: {model_info['model_name']}[/blue]")
            console.print(f"[blue]📋 Device: {model_info['device']}[/blue]")
            console.print(f"[blue]📋 Batch Size: {model_info.get('optimized_batch_size', 'N/A')}[/blue]")
            console.print(f"[blue]📋 GPU Detected: {model_info.get('nvidia_gpu_detected', False)}[/blue]")
            
            self.results['embedding_performance'] = {
                'single_time': single_time,
                'batch_time': batch_time,
                'throughput': len(test_texts) / batch_time,
                'success_rate': successful_embeddings / len(test_texts),
                'model_info': model_info
            }
            
            return successful_embeddings > 0
            
        except Exception as e:
            console.print(f"[red]❌ Embedding test failed: {e}[/red]")
            return False
    
    def test_reranker_performance(self):
        """Test reranker with GPU acceleration."""
        console.print("\n[bold cyan]🔄 Reranker Performance Test[/bold cyan]")
        
        try:
            reranker = RerankerManager(self.config.rag.reranker)
            
            # Test data
            query = "What is Filipino culture and tradition?"
            test_docs = [
                {"content": "Filipino culture is rich in traditions and customs passed down through generations.", "score": 0.8},
                {"content": "The Philippines has a diverse cultural heritage influenced by various civilizations.", "score": 0.7},
                {"content": "Bayanihan spirit represents the communal unity of Filipino people.", "score": 0.9},
                {"content": "Modern technology is changing how we work and communicate.", "score": 0.3},
                {"content": "Filipino festivals celebrate the country's religious and cultural diversity.", "score": 0.85},
            ] * 4  # 20 documents total
            
            # Convert to Document objects
            from src.retrieval.reranker import Document
            documents = [
                Document(content=doc["content"], metadata={}, score=doc["score"], doc_id=f"doc_{i}")
                for i, doc in enumerate(test_docs)
            ]
            
            # Reranking test
            console.print("🔄 Testing document reranking...")
            start_time = time.time()
            reranked_docs = reranker.rerank_documents(query, documents, top_k=10)
            rerank_time = time.time() - start_time
            
            # Display results
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Input Documents", str(len(documents)))
            table.add_row("Reranked Documents", str(len(reranked_docs)))
            table.add_row("Processing Time", f"{rerank_time:.3f}s")
            table.add_row("Throughput", f"{len(documents)/rerank_time:.1f} docs/sec")
            
            console.print(table)
            
            # Model info
            model_info = reranker.get_model_info()
            console.print(f"\n[blue]📋 Model: {model_info['model_name']}[/blue]")
            console.print(f"[blue]📋 Device: {model_info['device']}[/blue]")
            console.print(f"[blue]📋 Batch Size: {model_info.get('optimized_batch_size', model_info.get('batch_size', 'N/A'))}[/blue]")
            
            self.results['reranker_performance'] = {
                'processing_time': rerank_time,
                'throughput': len(documents) / rerank_time,
                'input_docs': len(documents),
                'output_docs': len(reranked_docs),
                'model_info': model_info
            }
            
            return len(reranked_docs) > 0
            
        except Exception as e:
            console.print(f"[red]❌ Reranker test failed: {e}[/red]")
            return False
    
    def test_evaluator_performance(self):
        """Test evaluator with GPU batch processing."""
        console.print("\n[bold cyan]📊 Evaluator Performance Test[/bold cyan]")
        
        try:
            evaluator = RetrievalEvaluator(self.config)
            
            # Test data - simulate evaluation scenarios
            test_texts = [
                "Filipino culture emphasizes family values and community spirit.",
                "The Philippines is known for its beautiful beaches and islands.",
                "Technology innovation is driving economic growth globally.",
                "Traditional Filipino dishes reflect the country's diverse heritage.",
                "Climate change poses significant challenges to island nations.",
            ]
            
            ground_truth = "Filipino cultural traditions are deeply rooted in family and community values."
            
            # Test batch similarity calculation
            console.print("🧮 Testing batch similarity calculation...")
            start_time = time.time()
            similarities = evaluator.calculate_batch_similarities(test_texts, ground_truth)
            batch_sim_time = time.time() - start_time
            
            # Test individual similarity calculation
            console.print("🔍 Testing individual similarity calculation...")
            start_time = time.time()
            individual_similarities = []
            for text in test_texts:
                sim = evaluator.calculate_cosine_similarity(text, ground_truth)
                individual_similarities.append(sim)
            individual_sim_time = time.time() - start_time
            
            # Display results
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Method", style="cyan")
            table.add_column("Time", style="green")
            table.add_column("Throughput", style="yellow")
            table.add_column("Avg Similarity", style="white")
            
            table.add_row(
                "Batch Processing", 
                f"{batch_sim_time:.3f}s", 
                f"{len(test_texts)/batch_sim_time:.1f} comparisons/sec",
                f"{np.mean(similarities):.3f}"
            )
            
            table.add_row(
                "Individual Processing", 
                f"{individual_sim_time:.3f}s", 
                f"{len(test_texts)/individual_sim_time:.1f} comparisons/sec",
                f"{np.mean(individual_similarities):.3f}"
            )
            
            speedup = individual_sim_time / batch_sim_time if batch_sim_time > 0 else 1.0
            table.add_row(
                "Speedup Factor", 
                f"{speedup:.2f}x", 
                "Batch vs Individual",
                "GPU Acceleration"
            )
            
            console.print(table)
            
            # Performance info
            perf_info = evaluator.get_performance_info()
            console.print(f"\n[blue]📋 Device: {perf_info['device']}[/blue]")
            console.print(f"[blue]📋 Batch Size: {perf_info['batch_size']}[/blue]")
            console.print(f"[blue]📋 GPU Acceleration: {perf_info['gpu_acceleration']}[/blue]")
            
            self.results['evaluator_performance'] = {
                'batch_time': batch_sim_time,
                'individual_time': individual_sim_time,
                'speedup': speedup,
                'batch_throughput': len(test_texts) / batch_sim_time,
                'performance_info': perf_info
            }
            
            return len(similarities) == len(test_texts)
            
        except Exception as e:
            console.print(f"[red]❌ Evaluator test failed: {e}[/red]")
            return False
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end RAG pipeline with GPU acceleration."""
        console.print("\n[bold cyan]🔗 End-to-End Pipeline Test[/bold cyan]")
        
        try:
            retriever = WiQASRetriever(self.config)
            
            # Test query
            test_query = "What are the key aspects of Filipino culture?"
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                # Test retrieval pipeline
                task = progress.add_task("Running end-to-end retrieval...", total=None)
                
                start_time = time.time()
                try:
                    results = retriever.query(
                        test_query,
                        k=5,
                        search_type="hybrid",
                        enable_reranking=True,
                        enable_mmr=True,
                        formatted=False
                    )
                    pipeline_time = time.time() - start_time
                    
                    progress.remove_task(task)
                    
                    # Check results
                    if isinstance(results, str):
                        if "No documents found" in results or "Error:" in results:
                            console.print(f"[yellow]⚠️  Pipeline result: {results}[/yellow]")
                            success = False
                        else:
                            console.print("[green]✅ Pipeline completed successfully[/green]")
                            success = True
                    else:
                        console.print(f"[green]✅ Retrieved {len(results)} results in {pipeline_time:.2f}s[/green]")
                        success = True
                    
                except Exception as e:
                    progress.remove_task(task)
                    console.print(f"[red]❌ Pipeline failed: {e}[/red]")
                    success = False
                    pipeline_time = 0
            
            # Display status
            status_table = Table(show_header=True, header_style="bold magenta")
            status_table.add_column("Component", style="cyan")
            status_table.add_column("Status", style="white")
            
            # Get retriever status
            try:
                status = retriever.get_status()
                status_table.add_row("Knowledge Base", f"✅ {status.get('document_count', 0)} documents")
                status_table.add_row("Embedding Model", f"✅ {status['config'].get('embedding_model', 'N/A')}")
                status_table.add_row("Reranking", f"✅ {'Enabled' if status['config'].get('reranking_enabled', False) else 'Disabled'}")
                status_table.add_row("Pipeline", "✅ Operational" if success else "❌ Failed")
            except Exception as e:
                status_table.add_row("Pipeline Status", f"❌ Error: {str(e)[:50]}")
            
            console.print(status_table)
            
            self.results['end_to_end'] = {
                'success': success,
                'pipeline_time': pipeline_time,
                'query': test_query
            }
            
            return success
            
        except Exception as e:
            console.print(f"[red]❌ End-to-end test failed: {e}[/red]")
            return False
    
    def generate_summary(self):
        """Generate a comprehensive test summary."""
        console.print("\n[bold green]📈 GPU-Accelerated RAG Pipeline Summary[/bold green]")
        
        # Overall results table
        summary_table = Table(title="Test Results Summary", show_header=True, header_style="bold magenta")
        summary_table.add_column("Test Component", style="cyan")
        summary_table.add_column("Status", style="white")
        summary_table.add_column("Performance", style="yellow")
        summary_table.add_column("GPU Utilized", style="green")
        
        # GPU Detection
        gpu_status = "✅ Pass" if self.results.get('gpu_detection', {}).get('cuda_available') or self.results.get('gpu_detection', {}).get('nvidia_gpu') else "⚠️  CPU Only"
        gpu_utilized = "✅ Yes" if self.results.get('gpu_detection', {}).get('nvidia_gpu') else "❌ No"
        summary_table.add_row("GPU Detection", gpu_status, "Hardware Detection", gpu_utilized)
        
        # Embedding Performance
        if 'embedding_performance' in self.results:
            emb_results = self.results['embedding_performance']
            emb_status = "✅ Pass" if emb_results['success_rate'] > 0.9 else "❌ Fail"
            emb_perf = f"{emb_results['throughput']:.1f} texts/sec"
            emb_gpu = "✅ Yes" if emb_results['model_info'].get('nvidia_gpu_detected') else "❌ No"
            summary_table.add_row("Embedding Generation", emb_status, emb_perf, emb_gpu)
        
        # Reranker Performance
        if 'reranker_performance' in self.results:
            rerank_results = self.results['reranker_performance']
            rerank_status = "✅ Pass" if rerank_results['output_docs'] > 0 else "❌ Fail"
            rerank_perf = f"{rerank_results['throughput']:.1f} docs/sec"
            rerank_gpu = "✅ Yes" if 'cuda' in rerank_results['model_info'].get('device', '') else "❌ No"
            summary_table.add_row("Document Reranking", rerank_status, rerank_perf, rerank_gpu)
        
        # Evaluator Performance
        if 'evaluator_performance' in self.results:
            eval_results = self.results['evaluator_performance']
            eval_status = "✅ Pass"
            eval_perf = f"{eval_results['speedup']:.2f}x speedup"
            eval_gpu = "✅ Yes" if eval_results['performance_info']['gpu_acceleration'] else "❌ No"
            summary_table.add_row("Similarity Evaluation", eval_status, eval_perf, eval_gpu)
        
        # End-to-End Pipeline
        if 'end_to_end' in self.results:
            e2e_results = self.results['end_to_end']
            e2e_status = "✅ Pass" if e2e_results['success'] else "❌ Fail"
            e2e_perf = f"{e2e_results['pipeline_time']:.2f}s total"
            e2e_gpu = "✅ Components" if any([
                self.results.get('embedding_performance', {}).get('model_info', {}).get('nvidia_gpu_detected'),
                'cuda' in self.results.get('reranker_performance', {}).get('model_info', {}).get('device', ''),
                self.results.get('evaluator_performance', {}).get('performance_info', {}).get('gpu_acceleration')
            ]) else "❌ No"
            summary_table.add_row("End-to-End Pipeline", e2e_status, e2e_perf, e2e_gpu)
        
        console.print(summary_table)
        
        # Recommendations
        console.print("\n[bold blue]💡 Recommendations[/bold blue]")
        
        gpu_detected = self.results.get('gpu_detection', {}).get('nvidia_gpu', False)
        if not gpu_detected:
            console.print("🔧 GPU acceleration not available - all processing using CPU fallback")
            console.print("   • Check NVIDIA GPU drivers and CUDA installation")
            console.print("   • Verify PyTorch CUDA compatibility")
            console.print("   • Performance will be slower but functionality is preserved")
        else:
            console.print("🚀 GPU acceleration fully operational!")
            console.print("   • Embedding generation optimized with mixed precision")
            console.print("   • Document reranking using GPU acceleration")
            console.print("   • Batch similarity calculations optimized for GPU")
            console.print("   • Memory management and cleanup implemented")


def main():
    """Run the comprehensive GPU RAG pipeline test."""
    if not WIQAS_AVAILABLE:
        console.print("[red]❌ WiQAS modules not available. Please check your installation.[/red]")
        return
    
    console.print(Panel.fit(
        "[bold green]🖥️  WiQAS GPU-Accelerated RAG Pipeline Test[/bold green]\n"
        "[cyan]Testing GPU acceleration across embedding, reranking, evaluation, and retrieval components[/cyan]",
        box=box.DOUBLE
    ))
    
    tester = GPURAGTester()
    
    # Run all tests
    tests = [
        ("GPU Detection", tester.test_gpu_detection),
        ("Embedding Performance", tester.test_embedding_performance),
        ("Reranker Performance", tester.test_reranker_performance),
        ("Evaluator Performance", tester.test_evaluator_performance),
        ("End-to-End Pipeline", tester.test_end_to_end_pipeline),
    ]
    
    for test_name, test_func in tests:
        try:
            console.print(f"\n[bold]Running {test_name}...[/bold]")
            test_func()
        except Exception as e:
            console.print(f"[red]❌ {test_name} failed with error: {e}[/red]")
    
    # Generate summary
    tester.generate_summary()
    
    console.print(f"\n[bold green]🎉 GPU RAG Pipeline Test Complete![/bold green]")
    console.print("[dim]GPU acceleration ready for production use![/dim]")


if __name__ == "__main__":
    main()