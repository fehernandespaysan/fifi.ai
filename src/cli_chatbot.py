"""
CLI Chatbot for Fifi.ai

Interactive command-line chatbot using RAG engine.
Features rich terminal UI, streaming responses, and conversation management.

Usage:
    python -m src.cli_chatbot

Commands:
    /help     - Show help
    /clear    - Clear conversation history
    /history  - Show conversation history
    /stats    - Show statistics
    /exit     - Exit the chatbot
"""

import sys
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.status import Status
from rich.table import Table

from src.blog_loader import BlogLoader
from src.config import get_config
from src.embeddings_manager import EmbeddingsManager
from src.logger import get_logger
from src.rag_engine import RAGEngine

logger = get_logger(__name__)


class CLIChatbot:
    """
    Interactive CLI chatbot using RAG engine.

    Provides a rich terminal interface with:
    - Streaming responses
    - Conversation history
    - Commands for management
    - Performance statistics
    """

    def __init__(self):
        """Initialize the CLI chatbot."""
        self.console = Console()
        self.config = get_config()

        # Initialize components
        self.embeddings_manager: Optional[EmbeddingsManager] = None
        self.rag_engine: Optional[RAGEngine] = None

        # State
        self.running = False

        logger.info("CLIChatbot initialized")

    def initialize_rag_engine(self) -> bool:
        """
        Initialize the RAG engine with embeddings.

        Returns:
            True if successful, False otherwise
        """
        try:
            with Status("[bold green]Initializing RAG engine...", console=self.console):
                # Initialize embeddings manager
                self.embeddings_manager = EmbeddingsManager()

                # Try to load existing index
                index_loaded = self.embeddings_manager.load()

                if not index_loaded:
                    self.console.print(
                        "[yellow]No existing index found. Building index...[/yellow]"
                    )

                    # Load blogs
                    loader = BlogLoader()
                    blogs = loader.load_all_blogs()

                    if not blogs:
                        self.console.print(
                            "[red]❌ No blog posts found. Please add blog posts to the blogs/ directory.[/red]"
                        )
                        return False

                    # Generate embeddings
                    self.console.print(
                        f"[cyan]Processing {len(blogs)} blog posts...[/cyan]"
                    )
                    self.embeddings_manager.add_documents(blogs)
                    self.embeddings_manager.save()

                    self.console.print("[green]✓ Index built and saved![/green]")

                # Initialize RAG engine
                self.rag_engine = RAGEngine(
                    embeddings_manager=self.embeddings_manager
                )

            self.console.print("[bold green]✓ RAG engine ready![/bold green]\n")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Failed to initialize: {str(e)}[/red]")
            logger.error(f"Initialization failed: {str(e)}", exc_info=True)
            return False

    def print_welcome(self):
        """Print welcome message."""
        welcome_text = f"""
# Welcome to {self.config.app_name}! 🤖

{self.config.welcome_message}

**Available Commands:**
- `/help`     - Show help
- `/clear`    - Clear conversation history
- `/history`  - Show conversation history
- `/stats`    - Show statistics
- `/exit`     - Exit the chatbot

**Tips:**
- Ask me anything about the topics in my knowledge base
- I'll provide answers with source citations
- Use commands to manage your session

Type your question and press Enter to begin!
"""
        panel = Panel(
            Markdown(welcome_text),
            title=f"[bold cyan]{self.config.app_name} RAG Chatbot[/bold cyan]",
            border_style="cyan",
        )
        self.console.print(panel)
        self.console.print()

    def handle_command(self, command: str) -> bool:
        """
        Handle special commands.

        Args:
            command: Command string (starts with /)

        Returns:
            True to continue, False to exit
        """
        command = command.lower().strip()

        if command == "/exit" or command == "/quit":
            return False

        elif command == "/help":
            self.show_help()

        elif command == "/clear":
            self.clear_history()

        elif command == "/history":
            self.show_history()

        elif command == "/stats":
            self.show_statistics()

        else:
            self.console.print(
                f"[yellow]Unknown command: {command}. Type /help for available commands.[/yellow]"
            )

        return True

    def show_help(self):
        """Show help message."""
        help_text = """
# Available Commands

- **`/help`**     - Show this help message
- **`/clear`**    - Clear conversation history
- **`/history`**  - Show conversation history
- **`/stats`**    - Show performance statistics
- **`/exit`**     - Exit the chatbot (or Ctrl+C)

# How to Use

1. Type your question and press Enter
2. Wait for the streaming response
3. Ask follow-up questions for context-aware conversations
4. Use commands to manage your session

# Examples

- "What is RAG?"
- "Tell me about vector databases"
- "How do I secure AI applications?"
"""
        panel = Panel(
            Markdown(help_text),
            title="[bold cyan]Help[/bold cyan]",
            border_style="cyan",
        )
        self.console.print(panel)

    def clear_history(self):
        """Clear conversation history."""
        if self.rag_engine:
            self.rag_engine.clear_history()
            self.console.print("[green]✓ Conversation history cleared![/green]")
        else:
            self.console.print("[yellow]No active session.[/yellow]")

    def show_history(self):
        """Show conversation history."""
        if not self.rag_engine:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        history = self.rag_engine.get_history()

        if not history:
            self.console.print("[yellow]No conversation history yet.[/yellow]")
            return

        table = Table(title="Conversation History", show_header=True, header_style="bold cyan")
        table.add_column("Role", style="dim", width=12)
        table.add_column("Message", style="")

        for msg in history:
            role_emoji = "👤 User" if msg.role == "user" else "🤖 Assistant"
            content_preview = (
                msg.content[:100] + "..."
                if len(msg.content) > 100
                else msg.content
            )
            table.add_row(role_emoji, content_preview)

        self.console.print(table)

    def show_statistics(self):
        """Show performance statistics."""
        if not self.rag_engine:
            self.console.print("[yellow]No active session.[/yellow]")
            return

        stats = self.rag_engine.get_statistics()
        embeddings_stats = self.embeddings_manager.get_statistics()

        # Create statistics table
        table = Table(title="Performance Statistics", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green")

        # RAG stats
        table.add_row("Total Queries", str(stats["total_queries"]))
        table.add_row("Total Tokens", f"{stats['total_tokens_used']:,}")
        table.add_row(
            "Avg Tokens/Query", f"{stats['avg_tokens_per_query']:.1f}"
        )
        table.add_row(
            "Avg Retrieval Time", f"{stats['avg_retrieval_time_ms']:.1f}ms"
        )
        table.add_row(
            "Avg Generation Time", f"{stats['avg_generation_time_ms']:.1f}ms"
        )

        # Embeddings stats
        table.add_row("─" * 30, "─" * 15)
        table.add_row("Index Size", str(embeddings_stats["total_vectors"]))
        table.add_row("Total Chunks", str(embeddings_stats["total_chunks"]))
        table.add_row("Embedding Model", embeddings_stats["embedding_model"])

        # Configuration
        table.add_row("─" * 30, "─" * 15)
        table.add_row("LLM Model", stats["model"])
        table.add_row("Temperature", str(stats["temperature"]))
        table.add_row("Top-K", str(stats["top_k"]))
        table.add_row("Min Relevance", str(stats["min_relevance_score"]))

        self.console.print(table)

        # Cost estimation
        if stats["total_queries"] > 0:
            avg_cost_per_1m_tokens = 0.375  # gpt-4o-mini average
            estimated_cost = (
                stats["total_tokens_used"] / 1_000_000
            ) * avg_cost_per_1m_tokens
            cost_per_query = estimated_cost / stats["total_queries"]

            self.console.print()
            self.console.print(
                f"[cyan]💰 Estimated Cost:[/cyan] [green]${estimated_cost:.4f}[/green] "
                f"[dim](${ cost_per_query:.4f}/query)[/dim]"
            )

    def process_query(self, query: str):
        """
        Process a user query and display response.

        Args:
            query: User's question
        """
        try:
            # Show user query
            self.console.print(f"\n[bold cyan]👤 You:[/bold cyan] {query}")
            self.console.print()

            # Stream response
            bot_name = self.config.app_name.split()[0]  # Use first word of app name
            self.console.print(f"[bold green]🤖 {bot_name}:[/bold green] ", end="")

            response_text = []
            for chunk in self.rag_engine.stream_query(query):
                self.console.print(chunk, end="")
                response_text.append(chunk)

            full_response = "".join(response_text)

            # Add spacing
            self.console.print("\n")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]⚠️  Response interrupted[/yellow]")
        except Exception as e:
            self.console.print(f"\n[red]❌ Error: {str(e)}[/red]")
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)

    def run(self):
        """Run the interactive chatbot."""
        # Initialize RAG engine
        if not self.initialize_rag_engine():
            return 1

        # Show welcome message
        self.print_welcome()

        self.running = True

        try:
            while self.running:
                try:
                    # Get user input
                    user_input = Prompt.ask(
                        "\n[bold cyan]👤 You[/bold cyan]"
                    ).strip()

                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.startswith("/"):
                        if not self.handle_command(user_input):
                            break
                        continue

                    # Process query
                    self.process_query(user_input)

                except KeyboardInterrupt:
                    self.console.print("\n")
                    confirm = Prompt.ask(
                        "[yellow]Exit chatbot?[/yellow]", choices=["y", "n"], default="n"
                    )
                    if confirm == "y":
                        break

                except EOFError:
                    break

        finally:
            self.show_goodbye()

        return 0

    def show_goodbye(self):
        """Show goodbye message."""
        if self.rag_engine:
            stats = self.rag_engine.get_statistics()
            self.console.print()
            self.console.print(
                f"[dim]Session stats: {stats['total_queries']} queries, "
                f"{stats['total_tokens_used']:,} tokens used[/dim]"
            )

        self.console.print(f"\n[bold cyan]👋 Thanks for using {self.config.app_name}! Goodbye![/bold cyan]\n")


def main():
    """Main entry point for CLI chatbot."""
    try:
        chatbot = CLIChatbot()
        return chatbot.run()
    except Exception as e:
        console = Console()
        console.print(f"[red]❌ Fatal error: {str(e)}[/red]")
        logger.error(f"Chatbot crashed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
