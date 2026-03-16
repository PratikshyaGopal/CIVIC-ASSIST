# wsgi.py — Vercel / gunicorn entry point
# Vercel's @vercel/python builder looks for a callable named 'app' in this file.
from app import app  # noqa: F401
