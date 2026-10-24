// Single source of truth for the backend URL.
// Override with VITE_API_BASE in frontend/.env (e.g. VITE_API_BASE=https://api.example.com).
export const API_BASE = (
  import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000"
).replace(/\/$/, "");

/** Turn FastAPI's `detail` (string or validation array) into a readable message. */
export function errorMessage(data, status) {
  const d = data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x) => x.msg || JSON.stringify(x)).join("; ");
  return `Request failed (${status}).`;
}

/** fetch + JSON + consistent errors. Throws Error with a user-facing message. */
export async function fetchJson(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, options);
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error("Could not reach the backend. Is it running?");
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const error = new Error(errorMessage(data, response.status));
    error.status = response.status;
    throw error;
  }
  return data;
}
