/**
 * lib/api.js — API helper for authenticated Flask requests
 *
 * Provides a wrapper around fetch() that automatically attaches
 * the JWT Authorization header for all requests to the Flask backend.
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

/**
 * Make an authenticated API request to the Flask backend.
 *
 * @param {string} endpoint - The API endpoint (e.g., "/api/rates")
 * @param {object} options - Fetch options (method, body, etc.)
 * @param {string} token - JWT token for authentication
 * @returns {Promise<object>} - Parsed JSON response
 */
export async function fetchWithAuth(endpoint, options = {}, token = null) {
  const url = `${API_URL}${endpoint}`;

  const headers = {
    "Content-Type": "application/json",
    ...(options.headers || {}),
  };

  // Attach JWT token if available
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  // Parse the response
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.error || `API error: ${response.status}`);
  }

  return data;
}

/**
 * Shorthand GET request
 */
export async function apiGet(endpoint, token) {
  return fetchWithAuth(endpoint, { method: "GET" }, token);
}

/**
 * Shorthand POST request
 */
export async function apiPost(endpoint, body, token) {
  return fetchWithAuth(
    endpoint,
    { method: "POST", body: JSON.stringify(body) },
    token
  );
}

/**
 * Shorthand PUT request
 */
export async function apiPut(endpoint, body, token) {
  return fetchWithAuth(
    endpoint,
    { method: "PUT", body: JSON.stringify(body) },
    token
  );
}

/**
 * Shorthand DELETE request
 */
export async function apiDelete(endpoint, token) {
  return fetchWithAuth(endpoint, { method: "DELETE" }, token);
}
