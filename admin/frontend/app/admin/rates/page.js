/**
 * app/admin/rates/page.js — Rates Management Page
 *
 * Data table listing all rate plans from the Flask API.
 * Features:
 * - Fetches GET /api/rates on mount
 * - Styled data table with columns: Origin, Destination, Carrier, Category, Tiers, Actions
 * - "Add New Rate" button
 * - Edit / Delete actions per row
 * - Delete confirmation modal
 * - Loading and empty states
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import Link from "next/link";
import { apiGet, apiDelete } from "../../../lib/api";

export default function RatesPage() {
  const { data: session } = useSession();
  const [rates, setRates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [deleteId, setDeleteId] = useState(null); // ID of rate to delete (shows modal)
  const [deleting, setDeleting] = useState(false);

  // Fetch all rates
  const fetchRates = useCallback(async () => {
    if (!session?.accessToken) return;

    try {
      setLoading(true);
      const data = await apiGet("/api/rates", session.accessToken);
      setRates(data.rates || []);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load rates");
    } finally {
      setLoading(false);
    }
  }, [session?.accessToken]);

  useEffect(() => {
    fetchRates();
  }, [fetchRates]);

  // Delete a rate plan
  const handleDelete = async () => {
    if (!deleteId) return;

    try {
      setDeleting(true);
      await apiDelete(`/api/rates/${deleteId}`, session.accessToken);
      setRates((prev) => prev.filter((r) => r._id !== deleteId));
      setDeleteId(null);
    } catch (err) {
      setError(err.message || "Failed to delete rate plan");
    } finally {
      setDeleting(false);
    }
  };

  // Carrier badge color
  const carrierColor = (carrier) => {
    const colors = {
      DHL: "#fbbf24",
      FedEx: "#818cf8",
      UPS: "#4ade80",
      Aramex: "#f87171",
    };
    return colors[carrier] || "#94a3b8";
  };

  return (
    <div>
      {/* Page header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
            Rate Plans
          </h1>
          <p className="mt-1 text-sm" style={{ color: "var(--color-text-secondary)" }}>
            Manage international shipping rates and weight tiers.
          </p>
        </div>

        <Link
          href="/admin/rates/new"
          className="btn-primary px-5 py-2.5 rounded-lg text-sm font-semibold text-white no-underline flex items-center gap-2"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Add New Rate
        </Link>
      </div>

      {/* Error message */}
      {error && (
        <div className="mb-4 p-3 rounded-lg text-sm"
             style={{ background: "rgba(239, 68, 68, 0.12)", color: "#f87171", border: "1px solid rgba(239, 68, 68, 0.2)" }}>
          {error}
        </div>
      )}

      {/* Loading state */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="flex flex-col items-center gap-3">
            <div className="spinner" style={{ width: "32px", height: "32px" }} />
            <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Loading rates...</p>
          </div>
        </div>
      ) : rates.length === 0 ? (
        /* Empty state */
        <div className="glass-card p-12 text-center">
          <div className="w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center"
               style={{ background: "rgba(99, 102, 241, 0.1)" }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round">
              <line x1="12" y1="1" x2="12" y2="23" />
              <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-1" style={{ color: "var(--color-text-primary)" }}>
            No Rate Plans Yet
          </h3>
          <p className="text-sm mb-4" style={{ color: "var(--color-text-muted)" }}>
            Create your first shipping rate plan to get started.
          </p>
          <Link
            href="/admin/rates/new"
            className="btn-primary inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold text-white no-underline"
          >
            Create First Rate
          </Link>
        </div>
      ) : (
        /* Data table */
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                  {["Origin", "Destination", "Carrier", "Category", "Unit", "Est. Days", "Tiers", "Actions"].map(
                    (col) => (
                      <th
                        key={col}
                        className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider"
                        style={{ color: "var(--color-text-muted)" }}
                      >
                        {col}
                      </th>
                    )
                  )}
                </tr>
              </thead>
              <tbody>
                {rates.map((rate) => (
                  <tr
                    key={rate._id}
                    className="table-row-hover"
                    style={{ borderBottom: "1px solid var(--color-border)" }}
                  >
                    <td className="px-5 py-3.5 font-medium" style={{ color: "var(--color-text-primary)" }}>
                      {rate.origin}
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {rate.destination}
                    </td>
                    <td className="px-5 py-3.5">
                      <span
                        className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold"
                        style={{
                          background: `${carrierColor(rate.carrier)}18`,
                          color: carrierColor(rate.carrier),
                          border: `1px solid ${carrierColor(rate.carrier)}30`,
                        }}
                      >
                        {rate.carrier}
                      </span>
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {rate.category}
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {rate.unit}
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {rate.estimated_days} days
                    </td>
                    <td className="px-5 py-3.5">
                      <span
                        className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium"
                        style={{ background: "rgba(99, 102, 241, 0.12)", color: "#818cf8" }}
                      >
                        {rate.tiers?.length || 0} tiers
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-2">
                        <Link
                          href={`/admin/rates/edit/${rate._id}`}
                          className="p-1.5 rounded-lg transition-colors no-underline"
                          style={{ color: "var(--color-text-muted)" }}
                          title="Edit"
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                          </svg>
                        </Link>
                        <button
                          onClick={() => setDeleteId(rate._id)}
                          className="p-1.5 rounded-lg transition-colors cursor-pointer"
                          style={{ color: "var(--color-text-muted)", background: "none", border: "none" }}
                          title="Delete"
                        >
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                            <polyline points="3 6 5 6 21 6" />
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          </svg>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="glass-card p-6 max-w-sm w-full mx-4 page-enter">
            <h3 className="text-lg font-semibold mb-2" style={{ color: "var(--color-text-primary)" }}>
              Delete Rate Plan?
            </h3>
            <p className="text-sm mb-5" style={{ color: "var(--color-text-secondary)" }}>
              This action cannot be undone. The rate plan will be permanently removed.
            </p>
            <div className="flex items-center gap-3 justify-end">
              <button
                onClick={() => setDeleteId(null)}
                className="px-4 py-2 rounded-lg text-sm font-medium cursor-pointer"
                style={{
                  background: "var(--color-bg-input)",
                  border: "1px solid var(--color-border)",
                  color: "var(--color-text-secondary)",
                }}
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={deleting}
                className="btn-danger px-4 py-2 rounded-lg text-sm font-semibold text-white border-none cursor-pointer disabled:opacity-50"
              >
                {deleting ? "Deleting..." : "Delete"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
