#!/bin/bash

echo "================================="
echo "PACS AI Agent - Server Startup"
echo "================================="
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "Copying .env.template to .env..."
    cp .env.template .env
    echo ""
    echo "⚠️  Please edit .env file:"
    echo "  - Add your SOLAR_API_KEY (optional for mock mode)"
    echo "  - Configure mock settings"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p data/dicom_storage
mkdir -p data/dicom_output
mkdir -p data/chroma_db
mkdir -p models
mkdir -p logs

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $python_version"

# Check if dependencies installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Dependencies not installed!"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo "✅ Dependencies OK"
echo ""

# Check Solar API key
if grep -q "your_solar_api_key_here" .env; then
    echo "⚠️  SOLAR_API_KEY not configured in .env"
    echo "Running in MOCK mode (no real API calls)"
    echo ""
fi

echo "🚀 Starting PACS AI Agent..."
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python api/main.py
