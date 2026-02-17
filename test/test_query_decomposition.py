"""
Quick test script for query decomposition functionality

Run this to verify the implementation works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.fusion import reciprocal_rank_fusion
from src.utilities.config import WiQASConfig, TimingBreakdown


def test_config():
    """Test that configuration is properly set up."""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    config = WiQASConfig()
    
    # Check query decomposition config exists
    assert hasattr(config.rag, 'query_decomposition'), "QueryDecompositionConfig not found in RAG config"
    
    decomp_config = config.rag.query_decomposition
    print(f"✅ Enabled: {decomp_config.enabled}")
    print(f"✅ Model: {decomp_config.model}")
    print(f"✅ Temperature: {decomp_config.temperature}")
    print(f"✅ Max sub-queries: {decomp_config.max_sub_queries}")
    print(f"✅ Enable fusion: {decomp_config.enable_fusion}")
    
    # Check timing breakdown has decomposition field
    timing = TimingBreakdown()
    assert hasattr(timing, 'query_decomposition_time'), "query_decomposition_time not found in TimingBreakdown"
    print(f"✅ Timing tracking available: query_decomposition_time")
    
    print("\n✅ Configuration test PASSED\n")
    return True


def test_timing_breakdown():
    """Test timing breakdown formatting."""
    print("=" * 60)
    print("Testing Timing Breakdown")
    print("=" * 60)
    
    timing = TimingBreakdown()
    timing.embedding_time = 0.15
    timing.search_time = 0.08
    timing.reranking_time = 0.12
    timing.query_decomposition_time = 0.25
    timing.total_time = 0.60
    
    # Get percentages
    percentages = timing.get_percentages()
    assert 'query_decomposition_percent' in percentages, "Missing query_decomposition_percent"
    
    print(f"✅ Query decomposition percentage: {percentages['query_decomposition_percent']:.2f}%")
    
    # Format summary
    summary = timing.format_timing_summary()
    assert 'query decomposition time' in summary, "Timing summary doesn't include decomposition"
    
    print("\n" + summary)
    print("\n✅ Timing breakdown test PASSED\n")
    return True


def test_fusion():
    """Test reciprocal rank fusion."""
    print("=" * 60)
    print("Testing Reciprocal Rank Fusion")
    print("=" * 60)
    
    # Create mock results from different queries
    results1 = [
        {"page_content": "Document A", "metadata": {"score": 0.9}},
        {"page_content": "Document B", "metadata": {"score": 0.8}},
        {"page_content": "Document C", "metadata": {"score": 0.7}},
    ]
    
    results2 = [
        {"page_content": "Document B", "metadata": {"score": 0.85}},  # Appears in both
        {"page_content": "Document D", "metadata": {"score": 0.75}},
        {"page_content": "Document A", "metadata": {"score": 0.70}},  # Appears in both
    ]
    
    # Fuse results
    fused = reciprocal_rank_fusion([results1, results2], k=60, deduplicate=True)
    
    print(f"Results from query 1: {len(results1)}")
    print(f"Results from query 2: {len(results2)}")
    print(f"Fused results: {len(fused)}")
    
    # Document B should rank highest (appears in top 2 of both lists)
    assert fused[0]["page_content"] == "Document B", "Fusion ranking incorrect"
    assert "rrf_score" in fused[0]["metadata"], "RRF score not added"
    assert fused[0]["metadata"]["query_appearances"] == 2, "Appearance count incorrect"
    
    print(f"\n✅ Top result: {fused[0]['page_content']}")
    print(f"   RRF Score: {fused[0]['metadata']['rrf_score']:.4f}")
    print(f"   Appearances: {fused[0]['metadata']['query_appearances']}")
    
    print("\n✅ Fusion test PASSED\n")
    return True


def test_query_decomposer_import():
    """Test that query decomposer can be imported."""
    print("=" * 60)
    print("Testing Query Decomposer Import")
    print("=" * 60)
    
    try:
        from src.retrieval.query_decomposer import QueryDecomposer
        print("✅ QueryDecomposer imported successfully")
        
        # Check class has required methods
        assert hasattr(QueryDecomposer, 'should_decompose'), "Missing should_decompose method"
        assert hasattr(QueryDecomposer, 'decompose'), "Missing decompose method"
        assert hasattr(QueryDecomposer, 'get_stats'), "Missing get_stats method"
        
        print("✅ All required methods present")
        print("\n✅ Import test PASSED\n")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import QueryDecomposer: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("QUERY DECOMPOSITION FEATURE TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Configuration", test_config),
        ("Timing Breakdown", test_timing_breakdown),
        ("Reciprocal Rank Fusion", test_fusion),
        ("Query Decomposer Import", test_query_decomposer_import),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test FAILED with error: {e}\n")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests PASSED! Query decomposition is ready to use.")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review errors above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
