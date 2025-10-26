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


def get_theme_css(theme="light"):
    """Generate theme-aware CSS based on selected theme."""
    # Theme colors
    themes = {
        "light": {
            "bg_primary": "#fafafa",
            "bg_secondary": "#ffffff",
            "bg_tertiary": "#f5f5f5",
            "text_primary": "#1a1a1a",
            "text_secondary": "#737373",
            "text_tertiary": "#a3a3a3",
            "border_light": "#f0f0f0",
            "border_medium": "#e5e5e5",
            "border_dark": "#d4d4d4",
            "accent_primary": "#1a1a1a",
            "accent_hover": "#fafafa",
            "shadow": "rgba(0, 0, 0, 0.05)",
        },
        "dark": {
            "bg_primary": "#1a1a1a",
            "bg_secondary": "#2a2a2a",
            "bg_tertiary": "#333333",
            "text_primary": "#fafafa",
            "text_secondary": "#b3b3b3",
            "text_tertiary": "#8a8a8a",
            "border_light": "#333333",
            "border_medium": "#404040",
            "border_dark": "#4a4a4a",
            "accent_primary": "#fafafa",
            "accent_hover": "#2a2a2a",
            "shadow": "rgba(255, 255, 255, 0.05)",
        }
    }

    colors = themes[theme]

    return f"""
<style>
    /* ===== CSS Variables for Theming ===== */
    :root {{
        --bg-primary: {colors['bg_primary']};
        --bg-secondary: {colors['bg_secondary']};
        --bg-tertiary: {colors['bg_tertiary']};
        --text-primary: {colors['text_primary']};
        --text-secondary: {colors['text_secondary']};
        --text-tertiary: {colors['text_tertiary']};
        --border-light: {colors['border_light']};
        --border-medium: {colors['border_medium']};
        --border-dark: {colors['border_dark']};
        --accent-primary: {colors['accent_primary']};
        --accent-hover: {colors['accent_hover']};
        --shadow: {colors['shadow']};
    }}

    /* ===== Global Styles ===== */
    .stApp {{
        background: var(--bg-primary);
        color: var(--text-primary);
        transition: background 0.3s ease, color 0.3s ease;
    }}

    /* Main content background */
    .main .block-container {{
        background: var(--bg-primary);
    }}

    /* Ensure all text elements use theme colors */
    p, span, div, label {{
        color: var(--text-primary);
    }}

    /* Streamlit specific text elements */
    .stMarkdown, .stText {{
        color: var(--text-primary) !important;
    }}

    /* Sidebar text elements */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
    }}

    [data-testid="stSidebar"] .stMarkdown {{
        color: var(--text-primary) !important;
    }}

    /* Hide Streamlit branding but keep important controls */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Keep the toolbar visible but hide deploy button */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}

    .stDeployButton {{
        visibility: hidden;
    }}

    /* Keep sidebar toggle visible and styled */
    button[kind="header"] {{
        visibility: visible !important;
        display: flex !important;
    }}

    /* Keep sidebar visible when collapsed - just narrow */
    [data-testid="stSidebar"][aria-expanded="false"] {{
        width: 60px !important;
        min-width: 60px !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] > div {{
        width: 60px !important;
    }}

    /* Style the sidebar toggle button */
    [data-testid="collapsedControl"] {{
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
        width: 48px !important;
        height: 48px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.15s ease !important;
    }}

    [data-testid="collapsedControl"]:hover {{
        background: var(--accent-hover) !important;
        border-color: var(--border-dark) !important;
    }}

    /* Make sure the icon is visible */
    [data-testid="collapsedControl"] svg {{
        fill: var(--text-primary) !important;
        width: 20px !important;
        height: 20px !important;
    }}

    /* Style the expand button inside sidebar */
    [data-testid="baseButton-header"] {{
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        padding: 0.5rem !important;
        margin: 0.5rem !important;
        transition: all 0.15s ease !important;
    }}

    [data-testid="baseButton-header"]:hover {{
        background: var(--accent-hover) !important;
        border-color: var(--border-dark) !important;
    }}

    [data-testid="baseButton-header"] svg {{
        fill: var(--text-primary) !important;
    }}

    /* Typography improvements */
    h1, h2, h3 {{
        letter-spacing: -0.02em;
        font-weight: 600;
        color: var(--text-primary);
    }}

    /* Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
    }}

    ::-webkit-scrollbar-track {{
        background: transparent;
    }}

    ::-webkit-scrollbar-thumb {{
        background: var(--border-medium);
        border-radius: 3px;
    }}

    ::-webkit-scrollbar-thumb:hover {{
        background: var(--border-dark);
    }}

    /* ===== Sidebar Styles ===== */
    [data-testid="stSidebar"] {{
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-light);
        width: 280px !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
        padding: 0;
    }}

    /* Logo and branding */
    .sidebar-logo {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1.5rem 1.5rem 1rem 1.5rem;
        border-bottom: 1px solid var(--bg-tertiary);
    }}

    .logo-box {{
        width: 32px;
        height: 32px;
        background: var(--accent-primary);
        color: var(--bg-primary);
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }}

    .logo-text {{
        display: flex;
        flex-direction: column;
        gap: 0.125rem;
    }}

    .logo-title {{
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1;
    }}

    .logo-subtitle {{
        font-size: 0.8125rem;
        color: var(--text-secondary);
        line-height: 1;
    }}

    /* Stats section */
    .stats-section {{
        padding: 1.5rem;
        border-bottom: 1px solid var(--bg-tertiary);
    }}

    .stat-item {{
        margin-bottom: 1.25rem;
    }}

    .stat-item:last-child {{
        margin-bottom: 0;
    }}

    .stat-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-tertiary);
        margin-bottom: 0.375rem;
        font-weight: 500;
    }}

    .stat-value {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        line-height: 1;
    }}

    /* Sidebar buttons */
    .stButton button {{
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-medium) !important;
        color: var(--text-primary) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
    }}

    .stButton button:hover {{
        background: var(--bg-secondary) !important;
        border-color: var(--border-dark) !important;
    }}

    /* ===== Main Content Area ===== */
    .main .block-container {{
        max-width: 680px;
        margin: 0 auto;
        padding: 3rem 2rem;
    }}

    /* Welcome screen */
    .welcome-header {{
        text-align: center;
        margin-bottom: 3rem;
    }}

    .welcome-title {{
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        margin-bottom: 0.5rem;
    }}

    .welcome-subtitle {{
        font-size: 0.9375rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }}

    /* Example questions - styled buttons */
    [data-testid="column"] {{
        gap: 0.75rem;
    }}

    /* Style the example question buttons */
    [data-testid="column"] .stButton button {{
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-light) !important;
        color: var(--text-primary) !important;
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
    }}

    [data-testid="column"] .stButton button:hover {{
        background: var(--bg-secondary) !important;
        border-color: var(--border-medium) !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px var(--shadow) !important;
    }}

    /* ===== Chat Messages ===== */
    .stChatMessage {{
        background: transparent !important;
        padding: 0 !important;
        margin-bottom: 2rem !important;
    }}

    .stChatMessage [data-testid="chatAvatarIcon-user"],
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {{
        width: 32px !important;
        height: 32px !important;
        border-radius: 50% !important;
    }}

    .stChatMessage [data-testid="chatAvatarIcon-user"] {{
        background: var(--bg-tertiary) !important;
    }}

    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {{
        background: var(--accent-primary) !important;
    }}

    .stChatMessage .stMarkdown {{
        font-size: 0.9375rem;
        line-height: 1.7;
        color: var(--text-primary);
    }}

    .stChatMessage .stMarkdown p {{
        margin-bottom: 1rem;
        color: var(--text-primary) !important;
    }}

    .stChatMessage .stMarkdown h1,
    .stChatMessage .stMarkdown h2,
    .stChatMessage .stMarkdown h3,
    .stChatMessage .stMarkdown h4 {{
        color: var(--text-primary) !important;
    }}

    .stChatMessage .stMarkdown a {{
        color: var(--accent-primary) !important;
        text-decoration: underline;
    }}

    .stChatMessage .stMarkdown code {{
        background: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        padding: 0.125rem 0.25rem;
        border-radius: 3px;
    }}

    .stChatMessage .stMarkdown pre {{
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 6px;
        padding: 1rem;
    }}

    .stChatMessage .stMarkdown pre code {{
        background: transparent !important;
    }}

    .stChatMessage .stMarkdown ul,
    .stChatMessage .stMarkdown ol {{
        color: var(--text-primary) !important;
    }}

    .stChatMessage .stMarkdown li {{
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem;
    }}

    /* ===== Sources Section ===== */
    .source-section {{
        border-top: 1px solid var(--bg-tertiary);
        padding-top: 1rem;
        margin-top: 1rem;
    }}

    .source-label {{
        font-size: 0.6875rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-tertiary);
        margin-bottom: 0.75rem;
        font-weight: 500;
    }}

    .source-card {{
        background: var(--bg-primary);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.15s ease;
    }}

    .source-card:hover {{
        background: var(--bg-secondary);
        border-color: var(--border-medium);
    }}

    .source-title {{
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }}

    .source-meta {{
        font-size: 0.75rem;
        color: var(--text-tertiary);
        margin-bottom: 0.75rem;
    }}

    .source-text {{
        font-size: 0.8125rem;
        color: var(--text-secondary);
        line-height: 1.6;
    }}

    /* ===== Chat Input Area ===== */
    /* Remove all container borders and backgrounds */
    .stChatInputContainer {{
        border: none !important;
        background: var(--bg-primary) !important;
        padding: 1rem 0 !important;
        max-width: 680px !important;
        margin: 0 auto !important;
    }}

    /* Fix the bottom chat input area - make it invisible */
    [data-testid="stBottom"] {{
        background: var(--bg-primary) !important;
        border: none !important;
    }}

    [data-testid="stBottom"] > div {{
        background: transparent !important;
        border: none !important;
    }}

    .stChatInput {{
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    .stChatInput > div {{
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
    }}

    /* Remove all nested container styling */
    .stChatInput div[data-baseweb="input"] {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    .stChatInput div[data-baseweb="input"] > div {{
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    /* Hide the submit button container background */
    .stChatInput button {{
        background: transparent !important;
        border: none !important;
    }}

    /* Only the textarea itself should have styling */
    .stChatInput textarea {{
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 8px !important;
        font-size: 0.9375rem !important;
        color: var(--text-primary) !important;
        padding: 0.75rem 1rem !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }}

    .stChatInput textarea:focus {{
        background: var(--bg-secondary) !important;
        border-color: var(--border-dark) !important;
        outline: none !important;
        box-shadow: 0 0 0 1px var(--border-dark) !important;
    }}

    .stChatInput textarea::placeholder {{
        color: var(--text-tertiary) !important;
    }}

    /* ===== Expander Styles ===== */
    .streamlit-expanderHeader {{
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        padding: 0.5rem 0 !important;
    }}

    .streamlit-expanderHeader:hover {{
        background: transparent !important;
    }}

    .streamlit-expanderHeader p {{
        color: var(--text-primary) !important;
    }}

    .streamlit-expanderHeader svg {{
        fill: var(--text-primary) !important;
    }}

    .streamlit-expanderContent {{
        border: none !important;
        padding: 0.5rem 0 !important;
        color: var(--text-primary) !important;
    }}

    .streamlit-expanderContent p,
    .streamlit-expanderContent div {{
        color: var(--text-primary) !important;
    }}

    /* ===== Metrics ===== */
    .metric-row {{
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--bg-tertiary);
    }}

    .metric-item {{
        flex: 1;
        font-size: 0.8125rem;
        color: var(--text-secondary);
    }}

    /* ===== Spinner ===== */
    .stSpinner > div {{
        border-top-color: var(--accent-primary) !important;
    }}

    /* ===== Info/Warning/Error boxes ===== */
    .stAlert {{
        background: var(--bg-primary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }}

    /* ===== Theme Toggle Button ===== */
    .theme-toggle {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.75rem 1rem;
        background: var(--bg-primary);
        border: 1px solid var(--border-medium);
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.15s ease;
        margin: 1rem 0;
    }}

    .theme-toggle:hover {{
        background: var(--bg-secondary);
        border-color: var(--border-dark);
    }}

    .theme-icon {{
        font-size: 1.25rem;
    }}

    .theme-label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-primary);
    }}

    /* ===== Mobile Responsiveness ===== */
    @media (max-width: 768px) {{
        /* Adjust main content padding on mobile */
        .main .block-container {{
            padding: 2rem 1rem;
            max-width: 100%;
        }}

        /* Welcome title smaller on mobile */
        .welcome-title {{
            font-size: 1.5rem;
        }}

        .welcome-subtitle {{
            font-size: 0.875rem;
        }}

        /* Stack example questions on mobile */
        [data-testid="column"] .stButton button {{
            padding: 0.875rem 1rem !important;
            font-size: 0.8125rem !important;
        }}

        /* Sidebar adjustments */
        [data-testid="stSidebar"] {{
            width: 100% !important;
        }}

        /* Logo adjustments */
        .sidebar-logo {{
            padding: 1rem;
        }}

        .logo-title {{
            font-size: 1rem;
        }}

        .logo-subtitle {{
            font-size: 0.75rem;
        }}

        /* Stats adjustments */
        .stats-section {{
            padding: 1rem;
        }}

        .stat-value {{
            font-size: 1.25rem;
        }}

        /* Source cards more compact */
        .source-card {{
            padding: 0.75rem;
        }}

        .source-title {{
            font-size: 0.8125rem;
        }}

        .source-text {{
            font-size: 0.75rem;
        }}

        /* Metrics stack vertically */
        .metric-row {{
            flex-direction: column;
            gap: 0.5rem;
        }}

        /* Touch-friendly buttons */
        .stButton button {{
            min-height: 44px !important;
            padding: 0.75rem 1rem !important;
        }}

        /* Chat messages more compact */
        .stChatMessage {{
            margin-bottom: 1.5rem !important;
        }}

        /* Input area */
        .stChatInput textarea {{
            font-size: 1rem !important; /* Prevent zoom on iOS */
        }}
    }}

    /* Tablet adjustments (769px - 1024px) */
    @media (min-width: 769px) and (max-width: 1024px) {{
        .main .block-container {{
            max-width: 90%;
            padding: 2.5rem 1.5rem;
        }}

        [data-testid="stSidebar"] {{
            width: 240px !important;
        }}
    }}
"""


# Apply theme-specific CSS (will be updated when theme changes)
def apply_theme():
    """Apply the current theme's CSS."""
    theme = st.session_state.get("theme", "light")
    st.markdown(get_theme_css(theme), unsafe_allow_html=True)


@st.cache_resource(ttl=3600)  # Cache for 1 hour, then refresh
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

    if "theme" not in st.session_state:
        st.session_state.theme = "light"  # Default to light theme


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

    # Theme toggle button
    st.sidebar.markdown("---")
    current_theme = st.session_state.get("theme", "light")
    theme_icon = "🌙" if current_theme == "light" else "☀️"
    theme_label = "Dark Mode" if current_theme == "light" else "Light Mode"

    if st.sidebar.button(f"{theme_icon} {theme_label}", use_container_width=True, key="theme_toggle"):
        # Toggle theme
        st.session_state.theme = "dark" if current_theme == "light" else "light"
        st.rerun()

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
    """Process query and get response with sources (non-streaming)."""
    with st.spinner("Searching knowledge base..."):
        response = rag_engine.query(query)
    return response


def process_query_streaming(rag_engine: RAGEngine, query: str):
    """
    Process query with streaming response.

    Returns a dictionary with:
        - answer: Full answer text
        - sources: List of sources
        - metadata: Response metadata
    """
    sources = []
    metadata = {}
    answer_placeholder = st.empty()
    full_answer = ""

    # Show searching indicator
    with st.spinner("🔍 Searching knowledge base..."):
        # Process the streaming query
        for chunk_data in rag_engine.query_stream(query):
            chunk_type = chunk_data["type"]

            if chunk_type == "sources":
                # Store sources for later display
                sources = chunk_data["content"]

            elif chunk_type == "chunk":
                # Stream text chunks
                full_answer += chunk_data["content"]
                answer_placeholder.markdown(full_answer + "▌")  # Show cursor

            elif chunk_type == "metadata":
                # Store metadata
                metadata = chunk_data["content"]

            elif chunk_type == "error":
                # Handle errors
                st.error(f"Error: {chunk_data['content']}")
                return None

    # Remove cursor and show final answer
    answer_placeholder.markdown(full_answer)

    return {
        "answer": full_answer,
        "sources": sources,
        "metadata": metadata,
    }


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Apply theme CSS
    apply_theme()

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

        # Get response with streaming
        with st.chat_message("assistant", avatar="✨"):
            response = process_query_streaming(rag_engine, prompt)

            if response:
                # Display sources
                if response["sources"]:
                    st.markdown('<div class="source-section">', unsafe_allow_html=True)
                    st.markdown(f'<div class="source-label">{len(response["sources"])} Sources</div>', unsafe_allow_html=True)

                    for i, source in enumerate(response["sources"], 1):
                        st.markdown(f"""
                        <div class="source-card">
                            <div class="source-title">{source.chunk.blog_title}</div>
                            <div class="source-meta">Score: {source.score:.3f} • Distance: {source.distance:.3f}</div>
                            <div class="source-text">{source.chunk.text[:300]}{'...' if len(source.chunk.text) > 300 else ''}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                # Display metrics
                metadata = response["metadata"]
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-item">⚡ {metadata.get('retrieval_time_ms', 0)}ms retrieval</div>
                    <div class="metric-item">🤖 {metadata.get('generation_time_ms', 0)}ms generation</div>
                    <div class="metric-item">🎯 {metadata.get('tokens_used', 0)} tokens</div>
                </div>
                """, unsafe_allow_html=True)

                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response["sources"]
                })

                # Update statistics
                st.session_state.total_queries += 1
                st.session_state.total_tokens += metadata.get('tokens_used', 0)


if __name__ == "__main__":
    main()
