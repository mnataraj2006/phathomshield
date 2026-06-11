/**
 * api.js — Central API base URL + resilient fetch utilities
 * Reads from VITE_API_URL env var at build time.
 * Falls back to localhost for local development.
 */
export const API_BASE = import.meta.env.VITE_API_URL || '/api';

/**
 * Wake the Render free-tier dyno with a lightweight GET /health ping.
 * Returns true if the server is up, false if it timed out.
 */
export async function pingBackend(timeoutMs = 8000) {
  try {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), timeoutMs);
    const res = await fetch(`${API_BASE}/health`, { signal: ctrl.signal });
    clearTimeout(tid);
    return res.ok;
  } catch {
    return false;
  }
}

/**
 * fetchWithRetry — wraps fetch() with:
 *  1. An optional warm-up ping (to wake Render free tier)
 *  2. Up to `maxRetries` retries on network/CORS failure
 *  3. Exponential back-off between retries
 *
 * Usage: same as fetch(), but pass options.onRetry(attempt, maxRetries)
 *        for UI feedback.
 */
export async function fetchWithRetry(url, options = {}, maxRetries = 2) {
  const { onRetry, ...fetchOpts } = options;
  let lastErr;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      const res = await fetch(url, fetchOpts);
      return res; // success — return even non-2xx so caller can inspect .ok
    } catch (err) {
      lastErr = err;
      if (attempt < maxRetries) {
        onRetry?.(attempt + 1, maxRetries);
        // Exponential back-off: 2s, 4s
        await new Promise(r => setTimeout(r, 2000 * Math.pow(2, attempt)));
      }
    }
  }
  throw lastErr;
}
