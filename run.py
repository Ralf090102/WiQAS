import asyncio
import os
import sys
from pathlib import Path

from src.utilities.config import get_config
from src.utilities.utils import get_colored_text, log_error, log_success, setup_logging

class WiQASCLI:
    """Interactive command line interface for WiQAS RAG system"""
    
    def __init__(self):
        self.config = get_config()
        log_success("WiQAS CLI initialized successfully")
            
    def print_banner(self):
        """Print welcome banner"""
        banner = """
╔═════════════════════════════════════════════════════════════════╗
║                                                                 ║
║        ░██       ░██ ░██  ░██████      ░███      ░██████        ║
║        ░██       ░██     ░██   ░██    ░██░██    ░██   ░██       ║
║        ░██  ░██  ░██ ░██░██     ░██  ░██  ░██  ░██              ║
║        ░██ ░████ ░██ ░██░██     ░██ ░█████████  ░████████       ║
║        ░██░██ ░██░██ ░██░██     ░██ ░██    ░██         ░██      ║
║        ░████   ░████ ░██ ░██   ░██  ░██    ░██  ░██   ░██       ║
║        ░███     ░███ ░██  ░██████   ░██    ░██   ░██████        ║
║                                ░██                              ║
║                                ░██                              ║
╚═════════════════════════════════════════════════════════════════╝
        """
        print(get_colored_text(banner, "cyan"))
        print(get_colored_text("🚀 Welcome to WiQAS - A RAG-Driven AI Assistant!", "green"))
        print(get_colored_text("   Your Intelligent Question Answering System", "blue"))
    
    def run(self):
        """Main application loop"""
        # Setup logging
        setup_logging()

        # Clear screen and show banner
        os.system("cls" if os.name == "nt" else "clear")
        self.print_banner()
        
        print(get_colored_text("\n" + "=" * 60, "yellow"))
        print(get_colored_text("System Status:", "white"))
        print(get_colored_text(f"✅ Configuration loaded", "green"))
        print(get_colored_text(f"✅ Logging initialized", "green"))
        print(get_colored_text(f"📁 Data directory: {self.config.system.storage.data_directory}", "cyan"))
        print(get_colored_text(f"🤖 LLM Model: {self.config.rag.llm.model}", "cyan"))
        print(get_colored_text(f"📊 Chunk size: {self.config.rag.chunking.chunk_size}", "cyan"))
        print(get_colored_text("=" * 60, "yellow"))
        
        print(get_colored_text("Press Ctrl+C to exit", "yellow"))
        
        try:
            input(get_colored_text("\nPress Enter to continue...", "cyan"))
        except KeyboardInterrupt:
            print(get_colored_text("\n\n👋 Goodbye! Thanks for using WiQAS!", "green"))

    
def main():
    """Entry point for the WiQAS CLI application"""
    try:
        cli = WiQASCLI()
        cli.run()
    except Exception as e:
        print(get_colored_text(f"❌ Failed to start WiQAS: {str(e)}", "red"))
        sys.exit(1)


if __name__ == "__main__":
    main()