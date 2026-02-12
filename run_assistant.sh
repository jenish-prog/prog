#!/bin/bash
source venv/bin/activate
export GEMINI_API_KEY="YOUR_API_KEY_HERE" # Replace with your actual key or export it in your shell
# Check if API key is set
if [ "$GEMINI_API_KEY" == "YOUR_API_KEY_HERE" ]; then
    echo "Please set your GEMINI_API_KEY in this script or your environment."
    echo "You can get a key from: https://aistudio.google.com/"
    # Continue anyway to let the script handle the warning/input
fi
python3 assistant.py
