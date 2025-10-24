#!/bin/bash
# Launcher script for Fifi.ai Streamlit Web UI

echo "🚀 Starting Fifi.ai Web Interface..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run Streamlit
streamlit run streamlit_app.py

# Or with custom port:
# streamlit run streamlit_app.py --server.port 8501
