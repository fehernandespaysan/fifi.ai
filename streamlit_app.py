"""
Fifi.ai Streamlit Web Interface

Beautiful web UI for the RAG chatbot using Streamlit.

Usage:
    streamlit run streamlit_app.py

Features:
- Modern minimalist design
- Chat interface with message history
- Source citations
- Statistics dashboard
- Conversation management
"""

import time
from datetime import datetime
from typing import List

import streamlit as st

from src.blog_loader import BlogLoader
from src.config import get_config
from src.embeddings_manager import EmbeddingsManager
from src.logger import setup_logging
from src.rag_engine import ConversationMessage, RAGEngine

# Setup logging
setup_logging(log_format="json")

# Get config for branding
config = get_config()

# Page configuration
st.set_page_config(
    page_title=f"{config.app_name} - RAG Chatbot",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",  # Always show sidebar on load
)

# Modern Minimalist CSS
st.markdown("""
<style>
    /* ===== Global Styles ===== */
    .stApp {
        background: #fafafa;
    }

    /* Hide Streamlit branding but keep important controls */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Keep the toolbar visible but hide deploy button */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    .stDeployButton {
        visibility: hidden;
    }

    /* Keep sidebar toggle visible and styled */
    button[kind="header"] {
        visibility: visible !important;
        display: flex !important;
    }

    /* Keep sidebar visible when collapsed - just narrow */
    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 60px !important;
        min-width: 60px !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] > div {
        width: 60px !important;
    }

    /* Style the sidebar toggle button */
    [data-testid="collapsedControl"] {
        background: white !important;
        border: 1px solid #f0f0f0 !important;
        border-radius: 6px !important;
        color: #1a1a1a !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
        width: 48px !important;
        height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="collapsedControl"]:hover {
        background: #fafafa !important;
        border-color: #d4d4d4 !important;
    }

    /* Make sure the icon is visible */
    [data-testid="collapsedControl"] svg {
        fill: #1a1a1a !important;
        width: 20px !important;
        height: 20px !important;
    }

    /* Style the expand button inside sidebar */
    [data-testid="baseButton-header"] {
        background: white !important;
        border: 1px solid #f0f0f0 !important;
        border-radius: 6px !important;
        color: #1a1a1a !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
        transition: all 0.15s ease !important;
    }

    [data-testid="baseButton-header"]:hover {
        background: #fafafa !important;
        border-color: #d4d4d4 !important;
    }

    [data-testid="baseButton-header"] svg {
        fill: #1a1a1a !important;
    }

    /* Typography improvements */
    h1, h2, h3 {
        letter-spacing: -0.02em;
        font-weight: 600;
        color: #1a1a1a;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
    }

    ::-webkit-scrollbar-track {
        background: transparent;
    }

    ::-webkit-scrollbar-thumb {
        background: #e5e5e5;
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #d4d4d4;
    }

    /* ===== Sidebar Styles ===== */
    [data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #f0f0f0;
        width: 280px !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        padding: 0;
    }

    /* Logo and branding */
    .sidebar-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        border-bottom: 1px solid #f5f5f5;
    }

    .logo-box {
        width: 32px;
        height: 32px;
        background: #1a1a1a;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }

    .logo-text {
        display: flex;
        flex-direction: column;
        gap: 0.125rem;
    }

    .logo-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.02em;
        line-height: 1;
    }

    .logo-subtitle {
        font-size: 0.8125rem;
        color: #737373;
        line-height: 1;
    }

    /* Stats section */
    .stats-section {
        padding: 1.5rem;
        border-bottom: 1px solid #f5f5f5;
    }

    .stat-item {
        margin-bottom: 1.25rem;
    }

    .stat-item:last-child {
        margin-bottom: 0;
    }

    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #a3a3a3;
        margin-bottom: 0.375rem;
        font-weight: 500;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a1a;
        line-height: 1;
    }

    /* Sidebar buttons */
    .stButton button {
        background: #fafafa !important;
        border: 1px solid #e5e5e5 !important;
        color: #1a1a1a !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }

    .stButton button:hover {
        background: white !important;
        border-color: #d4d4d4 !important;
    }

    /* ===== Main Content Area ===== */
    .main .block-container {
        max-width: 680px;
        margin: 0 auto;
        padding: 3rem 2rem;
    }

    /* Welcome screen */
    .welcome-header {
        text-align: center;
        margin-bottom: 3rem;
    }

    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }

    .welcome-subtitle {
        font-size: 0.9375rem;
        color: #737373;
        line-height: 1.6;
    }

    /* Example questions - styled buttons */
    [data-testid="column"] {
        gap: 0.75rem;
    }

    /* Style the example question buttons */
    [data-testid="column"] .stButton button {
        background: #fafafa !important;
        border: 1px solid #f0f0f0 !important;
        color: #1a1a1a !important;
        font-size: 0.875rem !important;
        font-weight: 400 !important;
        padding: 1rem 1.25rem !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
        line-height: 1.5 !important;
        min-height: auto !important;
        height: auto !important;
        white-space: normal !important;
    }

    [data-testid="column"] .stButton button:hover {
        background: white !important;
        border-color: #e5e5e5 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }

    /* ===== Chat Messages ===== */
    .stChatMessage {
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 2rem !important;
    }

    .stChatMessage [data-testid="chatAvatarIcon-user"],
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
    }

    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background: #f5f5f5 !important;
    }

    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background: #1a1a1a !important;
    }

    .stChatMessage .stMarkdown {
        font-size: 0.9375rem;
        line-height: 1.7;
        color: #1a1a1a;
    }

    .stChatMessage .stMarkdown p {
        margin-bottom: 1rem;
    }

    /* ===== Sources Section ===== */
    .source-section {
        border-top: 1px solid #f5f5f5;
        padding-top: 1rem;
        margin-top: 1rem;
    }

    .source-label {
        font-size: 0.6875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #a3a3a3;
        margin-bottom: 0.75rem;
        font-weight: 500;
    }

    .source-card {
        background: #fafafa;
        border: 1px solid #f0f0f0;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.15s ease;
    }

    .source-card:hover {
        background: white;
        border-color: #e5e5e5;
    }

    .source-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }

    .source-meta {
        font-size: 0.75rem;
        color: #a3a3a3;
        margin-bottom: 0.75rem;
    }

    .source-text {
        font-size: 0.8125rem;
        color: #737373;
        line-height: 1.6;
    }

    /* ===== Chat Input Area ===== */
    .stChatInputContainer {
        border-top: 1px solid #f0f0f0;
        padding: 1rem 0;
        max-width: 680px;
        margin: 0 auto;
    }

    .stChatInput textarea {
        background: #fafafa !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
        font-size: 0.9375rem !important;
        color: #1a1a1a !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.15s ease !important;
    }

    .stChatInput textarea:focus {
        background: white !important;
        border-color: #1a1a1a !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* ===== Expander Styles ===== */
    .streamlit-expanderHeader {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: #1a1a1a !important;
        padding: 0.5rem 0 !important;
    }

    .streamlit-expanderHeader:hover {
        background: transparent !important;
    }

    .streamlit-expanderContent {
        border: none !important;
        padding: 0.5rem 0 !important;
    }

    /* ===== Metrics ===== */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #f5f5f5;
    }

    .metric-item {
        flex: 1;
        font-size: 0.8125rem;
        color: #737373;
    }

    /* ===== Spinner ===== */
    .stSpinner > div {
        border-top-color: #1a1a1a !important;
    }

    /* ===== Info/Warning/Error boxes ===== */
    .stAlert {
        background: #fafafa !important;
        border: 1px solid #e5e5e5 !important;
        border-radius: 8px !important;
        color: #1a1a1a !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_engine():
    """Initialize the RAG engine (cached for performance)."""
    try:
        # Initialize embeddings manager
        embeddings_manager = EmbeddingsManager()

        # Try to load existing index
        index_loaded = embeddings_manager.load()

        if not index_loaded:
            st.warning("⚠️ No existing index found. Building index from blogs...")

            # Load blogs
            loader = BlogLoader()
            blogs = loader.load_all_blogs()

            if not blogs:
                st.error("❌ No blog posts found. Please add blog posts to the blogs/ directory.")
                return None

            # Generate embeddings
            with st.spinner(f"Processing {len(blogs)} blog posts..."):
                embeddings_manager.add_documents(blogs)
                embeddings_manager.save()

            st.success("✅ Index built and saved!")

        # Initialize RAG engine
        rag_engine = RAGEngine(embeddings_manager=embeddings_manager)

        return rag_engine

    except Exception as e:
        st.error(f"❌ Failed to initialize RAG engine: {str(e)}")
        return None


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0

    if "total_tokens" not in st.session_state:
        st.session_state.total_tokens = 0

    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now()

    if "selected_example" not in st.session_state:
        st.session_state.selected_example = None


def display_sidebar(rag_engine: RAGEngine):
    """Display minimal modern sidebar."""
    # Logo and branding
    st.sidebar.markdown("""
    <div class="sidebar-logo">
        <div class="logo-box">✨</div>
        <div class="logo-text">
            <div class="logo-title">Fifi</div>
            <div class="logo-subtitle">AI Engineering Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Get statistics
    stats = rag_engine.get_statistics()
    embeddings_stats = rag_engine.embeddings_manager.get_statistics()

    # Statistics section
    st.sidebar.markdown(f"""
    <div class="stats-section">
        <div class="stat-item">
            <div class="stat-label">Total Queries</div>
            <div class="stat-value">{stats["total_queries"]}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Tokens Used</div>
            <div class="stat-value">{stats["total_tokens_used"]:,}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Index Size</div>
            <div class="stat-value">{embeddings_stats["total_vectors"]}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Avg Response</div>
            <div class="stat-value">{stats['avg_generation_time_ms']:.0f}ms</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Controls section
    st.sidebar.markdown("")
    st.sidebar.markdown("")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = []
            rag_engine.clear_history()
            st.rerun()

    with col2:
        if st.button("Reset", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.rag_engine = None
            st.rerun()

    # Detailed stats (collapsible)
    with st.sidebar.expander("Detailed Stats", expanded=False):
        st.markdown(f"""
        **Model**: {stats['model']}
        **Temperature**: {stats['temperature']}
        **Top-K**: {stats['top_k']}
        **Chunks**: {embeddings_stats['total_chunks']}
        **Retrieval**: {stats['avg_retrieval_time_ms']:.0f}ms
        **Generation**: {stats['avg_generation_time_ms']:.0f}ms
        """)

        # Cost estimation
        if stats['total_tokens_used'] > 0:
            avg_cost_per_1m_tokens = 0.375  # gpt-4o-mini
            cost = (stats['total_tokens_used'] / 1_000_000) * avg_cost_per_1m_tokens
            st.markdown(f"**Estimated Cost**: ${cost:.4f}")

    # Session info
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    session_duration = (datetime.now() - st.session_state.session_start).total_seconds()
    st.sidebar.caption(f"Session duration: {session_duration/60:.1f} minutes")


def display_welcome_screen():
    """Display welcome screen with example questions."""
    config = get_config()

    st.markdown(f"""
    <div class="welcome-header">
        <h1 class="welcome-title">{config.welcome_title}</h1>
        <p class="welcome-subtitle">
            {config.welcome_message}
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Example questions - 2 columns with 2 questions each
    examples_left = [
        config.example_question_1,
        config.example_question_2,
    ]

    examples_right = [
        config.example_question_3,
        config.example_question_4,
    ]

    # Create 2-column layout using Streamlit columns
    col1, col2 = st.columns(2)

    # Left column - 2 questions
    with col1:
        for example in examples_left:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.selected_example = example

    # Right column - 2 questions
    with col2:
        for example in examples_right:
            if st.button(example, key=f"example_{hash(example)}", use_container_width=True):
                st.session_state.selected_example = example


def display_message_with_sources(role: str, content: str, sources=None):
    """Display a chat message with sources."""
    with st.chat_message(role, avatar="✨" if role == "assistant" else None):
        st.markdown(content)

        # Display sources if available
        if sources and len(sources) > 0:
            st.markdown('<div class="source-section">', unsafe_allow_html=True)
            st.markdown(f'<div class="source-label">{len(sources)} Sources</div>', unsafe_allow_html=True)

            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-title">{source.chunk.blog_title}</div>
                    <div class="source-meta">Score: {source.score:.3f} • Distance: {source.distance:.3f}</div>
                    <div class="source-text">{source.chunk.text[:300]}{'...' if len(source.chunk.text) > 300 else ''}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


def process_query_with_sources(rag_engine: RAGEngine, query: str):
    """Process query and get response with sources."""
    with st.spinner("Searching knowledge base..."):
        response = rag_engine.query(query)
    return response


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Initialize RAG engine (cached)
    if st.session_state.rag_engine is None:
        with st.spinner("Initializing RAG engine..."):
            st.session_state.rag_engine = initialize_rag_engine()

    rag_engine = st.session_state.rag_engine

    if rag_engine is None:
        st.error("Failed to initialize RAG engine. Please check your configuration.")
        return

    # Display sidebar
    display_sidebar(rag_engine)

    # Show welcome screen if no messages
    if len(st.session_state.messages) == 0:
        display_welcome_screen()

    # Handle example question selection
    prompt = None
    if st.session_state.selected_example:
        prompt = st.session_state.selected_example
        st.session_state.selected_example = None  # Reset after using

    # Display chat messages
    for message in st.session_state.messages:
        display_message_with_sources(
            message["role"],
            message["content"],
            message.get("sources")
        )

    # Chat input (if no example was selected)
    if not prompt:
        prompt = st.chat_input("Ask me anything...")

    # Process the prompt (from either example or chat input)
    if prompt:
        # Display user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        display_message_with_sources("user", prompt)

        # Get response
        with st.chat_message("assistant", avatar="✨"):
            response = process_query_with_sources(rag_engine, prompt)

            if response:
                # Display response
                st.markdown(response.answer)

                # Display sources
                if response.sources:
                    st.markdown('<div class="source-section">', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-label">{len(response.sources)} Sources</div>', unsafe_allow_html=True)

                    for i, source in enumerate(response.sources, 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{source.chunk.blog_title}</div>
                            <div class="source-meta">Score: {source.score:.3f} • Distance: {source.distance:.3f}</div>
                            <div class="source-text">{source.chunk.text[:300]}{'...' if len(source.chunk.text) > 300 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                # Display metrics
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-item">⚡ {response.retrieval_time_ms}ms retrieval</div>
                    <div class="metric-item">🤖 {response.generation_time_ms}ms generation</div>
                    <div class="metric-item">🎯 {response.tokens_used} tokens</div>
                </div>
                """, unsafe_allow_html=True)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources
                })

                # Update statistics
                st.session_state.total_queries += 1
                st.session_state.total_tokens += response.tokens_used


if __name__ == "__main__":
    main()
