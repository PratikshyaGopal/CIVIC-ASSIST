# api/index.py — Vercel serverless entry point
# Vercel's @vercel/python builder looks for a callable named 'app' in this file.
import sys
import os

# Make sure the project root is on the path so we can import app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: F401
