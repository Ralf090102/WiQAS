import argparse
import importlib
import subprocess
import sys
from pathlib import Path

# ================================
# DEPENDENCY DEFINITIONS
# ================================

DEPENDENCIES = {
    "base": [
        "ollama>=0.5.3",
        "langchain",
        "langchain-community",
        "langchain-ollama",
        "langchain-text-splitters",
        "sentence-transformers",
        "chromadb",
        "chardet",
        "rank_bm25",
        "numpy",
        "transformers",
        "accelerate",
        "tqdm",
        "rich",
        "psutil",
    ],
    "docs": ["pypdf", "pandas", "openpyxl"],
    "dev": ["pytest", "pytest-asyncio", "pytest-cov", "black", "ruff", "pre-commit", "typer"],
    "ocr": ["pytesseract", "Pillow", "easyocr"],
    "gpu": ["torch>=2.7.1+cu121", "torchvision>=0.22.0+cu121"],
}

# Model configurations for testing
MODEL_CONFIGS = {
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "type": "embedding",
        "framework": "sentence-transformers",
        "test_text": "Kumusta! This is a test in Filipino and English.",
        "expected_dim": 1024,
        "dependencies": ["sentence-transformers", "transformers", "torch"],
    },
    "ollama": {
        "name": "mistral:latest",
        "type": "llm",
        "framework": "ollama",
        "test_text": "Hello, how are you?",
        "dependencies": ["ollama"],
    },
}

# Import test mappings
IMPORT_TESTS = {
    "sentence-transformers": "sentence_transformers",
    "langchain": "langchain",
    "langchain-community": "langchain_community",
    "langchain-ollama": "langchain_ollama",
    "langchain-text-splitters": "langchain_text_splitters",
    "chromadb": "chromadb",
    "ollama": "ollama",
    "transformers": "transformers",
    "torch": "torch",
    "numpy": "numpy",
    "rich": "rich",
    "tqdm": "tqdm",
    "pypdf": "pypdf",
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "pytest": "pytest",
    "black": "black",
    "ruff": "ruff",
    "rank_bm25": "rank_bm25",
    "chardet": "chardet",
    "accelerate": "accelerate",
    "psutil": "psutil",
}


# ================================
# UTILITY FUNCTIONS
# ================================


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n📦 {title}")
    print("-" * 40)


def print_success(message: str) -> None:
    """Print a success message."""
    print(f"✅ {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    print(f"❌ {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    print(f"⚠️  {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    print(f"ℹ️  {message}")


# ================================
# DEPENDENCY MANAGEMENT
# ================================


def install_dependencies(groups: list[str], upgrade: bool = False) -> bool:
    """
    Install dependency groups.

    Args:
        groups: List of dependency groups to install
        upgrade: Whether to upgrade existing packages

    Returns:
        True if successful, False otherwise
    """
    print_section("Installing Dependencies")

    packages = []
    for group in groups:
        if group in DEPENDENCIES:
            packages.extend(DEPENDENCIES[group])
            print_info(f"Added {group} dependencies: {len(DEPENDENCIES[group])} packages")
        else:
            print_warning(f"Unknown dependency group: {group}")

    if not packages:
        print_error("No packages to install")
        return False

    # Remove duplicates
    unique_packages = list(dict.fromkeys(packages))
    print_info(f"Total unique packages to install: {len(unique_packages)}")

    cmd = [sys.executable, "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    cmd.extend(unique_packages)

    try:
        print_info("Starting installation...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_success("All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def install_from_requirements(file_path: str = "requirements.txt", upgrade: bool = False) -> bool:
    """
    Install dependencies from requirements.txt.

    Args:
        file_path: Path to requirements file
        upgrade: Whether to upgrade existing packages

    Returns:
        True if successful, False otherwise
    """
    print_section(f"Installing from {file_path}")

    req_file = Path(file_path)
    if not req_file.exists():
        print_error(f"Requirements file not found: {file_path}")
        return False

    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
    if upgrade:
        cmd.append("--upgrade")

    try:
        print_info(f"Installing from {file_path}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print_success("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Installation failed: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


# ================================
# IMPORT TESTING
# ================================


def test_imports(groups: list[str] | None = None) -> dict[str, bool]:
    """
    Test if packages can be imported.

    Args:
        groups: List of dependency groups to test, None for all

    Returns:
        Dictionary mapping package names to import success status
    """
    print_section("Testing Imports")

    if groups:
        packages_to_test = []
        for group in groups:
            if group in DEPENDENCIES:
                packages_to_test.extend(DEPENDENCIES[group])
    else:
        packages_to_test = [pkg for group in DEPENDENCIES.values() for pkg in group]

    package_names = []
    for pkg in packages_to_test:
        name = pkg.split(">=")[0].split("==")[0].split("<")[0].split(">")[0]
        package_names.append(name)

    package_names = list(set(package_names))

    results = {}

    for package in package_names:
        import_name = IMPORT_TESTS.get(package, package)

        try:
            importlib.import_module(import_name)
            print_success(f"{package} -> {import_name}")
            results[package] = True
        except ImportError:
            print_error(f"{package} -> {import_name}")
            results[package] = False

    successful = sum(results.values())
    total = len(results)
    print_info(f"Import test results: {successful}/{total} successful")

    return results


# ================================
# MODEL TESTING
# ================================


def test_model_loading(model_names: list[str] | None = None) -> dict[str, bool]:
    """
    Test loading and basic functionality of models.

    Args:
        model_names: List of model names to test, None for all

    Returns:
        Dictionary mapping model names to test success status
    """
    print_section("Testing Model Loading")

    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())

    results = {}

    for model_name in model_names:
        if model_name not in MODEL_CONFIGS:
            print_warning(f"Unknown model: {model_name}")
            results[model_name] = False
            continue

        config = MODEL_CONFIGS[model_name]
        print_info(f"Testing {model_name} ({config['type']})...")

        missing_deps = []
        for dep in config["dependencies"]:
            import_name = IMPORT_TESTS.get(dep, dep)
            try:
                importlib.import_module(import_name)
            except ImportError:
                missing_deps.append(dep)

        if missing_deps:
            print_error(f"Missing dependencies for {model_name}: {missing_deps}")
            results[model_name] = False
            continue

        # Test model loading
        try:
            if config["framework"] == "sentence-transformers":
                success = _test_sentence_transformer(config)
            elif config["framework"] == "ollama":
                success = _test_ollama_model(config)
            else:
                print_warning(f"Unknown framework: {config['framework']}")
                success = False

            results[model_name] = success

        except Exception as e:
            print_error(f"Error testing {model_name}: {e}")
            results[model_name] = False

    successful = sum(results.values())
    total = len(results)
    print_info(f"Model test results: {successful}/{total} successful")

    return results


def _test_sentence_transformer(config: dict) -> bool:
    """Test a sentence transformer model."""
    try:
        sys.path.insert(0, "src")
        from src.retrieval.embeddings import create_embedding_manager
        from src.utilities.config import get_config

        # Create embedding manager
        wiqas_config = get_config()
        embedding_manager = create_embedding_manager(wiqas_config)

        # Test embedding generation
        embedding = embedding_manager.encode_single(config["test_text"])

        if embedding is None:
            print_error("Failed to generate embedding")
            return False

        if len(embedding) != config["expected_dim"]:
            print_error(f"Unexpected embedding dimension: {len(embedding)} != {config['expected_dim']}")
            return False

        print_success(f"Model loaded, embedding dimension: {len(embedding)}")
        return True

    except Exception as e:
        print_error(f"Sentence transformer test failed: {e}")
        return False


def _test_ollama_model(config: dict) -> bool:
    """Test an Ollama model."""
    try:
        import ollama

        try:
            models = ollama.list()
            print_info("Ollama service is running")
        except Exception:
            print_error("Ollama service not running or not accessible")
            return False

        model_name = config["name"]
        available_models = [m["name"] for m in models.get("models", [])]

        if model_name not in available_models:
            print_warning(f"Model {model_name} not found in Ollama")
            print_info(f"Available models: {available_models}")
            return False

        # Test generation
        response = ollama.generate(model=model_name, prompt=config["test_text"])

        if response and response.get("response"):
            print_success(f"Ollama model {model_name} working")
            return True
        else:
            print_error(f"No response from Ollama model {model_name}")
            return False

    except Exception as e:
        print_error(f"Ollama test failed: {e}")
        return False


# ================================
# SYSTEM HEALTH CHECK
# ================================


def system_health_check() -> dict[str, any]:
    """
    Comprehensive system health check.

    Returns:
        Dictionary with health check results
    """
    print_section("System Health Check")

    health = {"python_version": sys.version, "pip_version": None, "gpu_available": False, "disk_space": None, "memory": None}

    # Check pip version
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, check=True)
        health["pip_version"] = result.stdout.strip()
        print_success(f"Pip: {health['pip_version']}")
    except Exception:
        print_error("Pip not available")

    # Check GPU availability
    try:
        import torch

        if torch.cuda.is_available():
            health["gpu_available"] = True
            gpu_name = torch.cuda.get_device_name(0)
            print_success(f"GPU available: {gpu_name}")
        else:
            print_info("No GPU available, using CPU")
    except ImportError:
        print_info("PyTorch not available, cannot check GPU")

    # Check disk space
    try:
        import psutil

        disk_usage = psutil.disk_usage(".")
        free_gb = disk_usage.free / (1024**3)
        health["disk_space"] = free_gb
        print_success(f"Free disk space: {free_gb:.1f} GB")

        if free_gb < 5:
            print_warning("Low disk space (< 5GB)")
    except ImportError:
        print_info("psutil not available, cannot check disk space")

    # Check memory
    try:
        import psutil

        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        health["memory"] = available_gb
        print_success(f"Available memory: {available_gb:.1f} GB")

        if available_gb < 4:
            print_warning("Low memory (< 4GB)")
    except ImportError:
        print_info("psutil not available, cannot check memory")

    print_success(f"Python: {sys.version.split()[0]}")

    return health


# ================================
# MAIN
# ================================


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="WiQAS Dependency Manager - Install, test, and validate dependencies and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Install base dependencies
  python set_dependencies.py --install base

  # Install from requirements.txt
  python set_dependencies.py --requirements

  # Test imports for base dependencies
  python set_dependencies.py --test-imports base

  # Test model loading
  python set_dependencies.py --test-models bge-m3

  # Full system check
  python set_dependencies.py --all

  # Install everything and test
  python set_dependencies.py --install base docs --test-imports --test-models --health-check
""",
    )

    # Installation options
    parser.add_argument("--install", nargs="+", metavar="GROUP", help="Install dependency groups: base, docs, dev, ocr, gpu")
    parser.add_argument("--requirements", action="store_true", help="Install from requirements.txt")
    parser.add_argument("--upgrade", action="store_true", help="Upgrade existing packages during installation")

    # Testing options
    parser.add_argument("--test-imports", nargs="*", metavar="GROUP", help="Test imports for dependency groups (no args = test all)")
    parser.add_argument("--test-models", nargs="*", metavar="MODEL", help="Test model loading (no args = test all)")

    # System options
    parser.add_argument("--health-check", action="store_true", help="Run system health check")
    parser.add_argument("--all", action="store_true", help="Run all operations (install base, test imports, test models, health check)")

    # Information options
    parser.add_argument("--list-groups", action="store_true", help="List available dependency groups")
    parser.add_argument("--list-models", action="store_true", help="List available models for testing")

    args = parser.parse_args()

    print_header("WiQAS Dependency Manager")
    print_info(f"Python: {sys.version.split()[0]}")
    print_info(f"Working directory: {Path.cwd()}")

    success = True

    # Handle information requests
    if args.list_groups:
        print_section("Available Dependency Groups")
        for group, packages in DEPENDENCIES.items():
            print(f"📦 {group}: {len(packages)} packages")
            for pkg in packages[:3]:  # Show first 3
                print(f"   - {pkg}")
            if len(packages) > 3:
                print(f"   ... and {len(packages) - 3} more")
        return

    if args.list_models:
        print_section("Available Models for Testing")
        for model, config in MODEL_CONFIGS.items():
            print(f"🤖 {model} ({config['type']})")
            print(f"   Framework: {config['framework']}")
            print(f"   Model: {config['name']}")
        return

    # Handle --all flag
    if args.all:
        args.install = ["base"]
        args.test_imports = []
        args.test_models = []
        args.health_check = True

    # Installation
    if args.install:
        success &= install_dependencies(args.install, args.upgrade)

    if args.requirements:
        success &= install_from_requirements(upgrade=args.upgrade)

    # Testing
    if args.test_imports is not None:
        results = test_imports(args.test_imports if args.test_imports else None)
        if not all(results.values()):
            success = False

    if args.test_models is not None:
        results = test_model_loading(args.test_models if args.test_models else None)
        if not all(results.values()):
            success = False

    # System health
    if args.health_check:
        health = system_health_check()

    # Final status
    print_header("Summary")
    if success:
        print_success("All operations completed successfully!")
        print_info("WiQAS is ready for use 🚀")
    else:
        print_error("Some operations failed")
        print_info("Check the output above for details")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
