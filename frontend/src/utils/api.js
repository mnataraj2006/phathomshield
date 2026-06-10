/**
 * api.js — Central API base URL configuration
 * Reads from VITE_API_URL env var at build time.
 * Falls back to localhost for local development.
 */
export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
