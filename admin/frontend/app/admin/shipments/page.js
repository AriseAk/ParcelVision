/**
 * app/admin/shipments/page.js — Shipments Management Page
 *
 * Data table listing all shipments from the Flask API.
 * Features:
 * - Fetches GET /api/shipments on mount
 * - Styled data table with status badges
 * - Search bar for filtering
 * - Create new shipment modal
 * - Delete confirmation modal
 * - Loading and empty states
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { apiGet, apiPost, apiDelete } from "../../../lib/api";

const CARRIERS = ["DHL", "FedEx", "UPS", "Aramex"];
const STATUSES = ["pending", "picked_up", "in_transit", "out_for_delivery", "delivered", "cancelled", "returned"];

const STATUS_COLORS = {
  pending:          { bg: "rgba(245, 158, 11, 0.12)", color: "#f59e0b", border: "rgba(245, 158, 11, 0.25)" },
  picked_up:        { bg: "rgba(99, 102, 241, 0.12)", color: "#818cf8", border: "rgba(99, 102, 241, 0.25)" },
  in_transit:       { bg: "rgba(59, 130, 246, 0.12)", color: "#60a5fa", border: "rgba(59, 130, 246, 0.25)" },
  out_for_delivery: { bg: "rgba(168, 85, 247, 0.12)", color: "#a78bfa", border: "rgba(168, 85, 247, 0.25)" },
  delivered:        { bg: "rgba(34, 197, 94, 0.12)",  color: "#4ade80", border: "rgba(34, 197, 94, 0.25)" },
  cancelled:        { bg: "rgba(239, 68, 68, 0.12)",  color: "#f87171", border: "rgba(239, 68, 68, 0.25)" },
  returned:         { bg: "rgba(156, 163, 175, 0.12)",color: "#9ca3af", border: "rgba(156, 163, 175, 0.25)" },
};

const CARRIER_COLORS = {
  DHL:    "#fbbf24",
  FedEx:  "#818cf8",
  UPS:    "#4ade80",
  Aramex: "#f87171",
};

const EMPTY_FORM = {
  tracking_number: "", sender_name: "", sender_address: "",
  receiver_name: "", receiver_address: "", origin: "", destination: "",
  carrier: "DHL", weight: "", unit: "KG", status: "pending", notes: "",
};

export default function ShipmentsPage() {
  const { data: session } = useSession();
  const [shipments, setShipments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [deleteId, setDeleteId] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [creating, setCreating] = useState(false);

  // Fetch shipments
  const fetchShipments = useCallback(async () => {
    if (!session?.accessToken) return;
    try {
      setLoading(true);
      const params = search ? `?search=${encodeURIComponent(search)}` : "";
      const data = await apiGet(`/api/shipments${params}`, session.accessToken);
      setShipments(data.shipments || []);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load shipments");
    } finally {
      setLoading(false);
    }
  }, [session?.accessToken, search]);

  useEffect(() => { fetchShipments(); }, [fetchShipments]);

  // Create shipment
  const handleCreate = async (e) => {
    e.preventDefault();
    try {
      setCreating(true);
      await apiPost("/api/shipments", form, session.accessToken);
      setShowCreate(false);
      setForm({ ...EMPTY_FORM });
      fetchShipments();
    } catch (err) {
      setError(err.message || "Failed to create shipment");
    } finally {
      setCreating(false);
    }
  };

  // Delete shipment
  const handleDelete = async () => {
    if (!deleteId) return;
    try {
      setDeleting(true);
      await apiDelete(`/api/shipments/${deleteId}`, session.accessToken);
      setShipments((prev) => prev.filter((s) => s._id !== deleteId));
      setDeleteId(null);
    } catch (err) {
      setError(err.message || "Failed to delete shipment");
    } finally {
      setDeleting(false);
    }
  };

  const formatStatus = (status) => status.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div>
      {/* Page header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
            Shipments
          </h1>
          <p className="mt-1 text-sm" style={{ color: "var(--color-text-secondary)" }}>
            Track and manage all shipments across carriers.
          </p>
        </div>

        <button
          id="create-shipment-btn"
          onClick={() => setShowCreate(true)}
          className="btn-primary px-5 py-2.5 rounded-lg text-sm font-semibold text-white border-none cursor-pointer flex items-center gap-2"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Shipment
        </button>
      </div>

      {/* Search bar */}
      <div className="mb-5">
        <input
          id="shipment-search"
          type="text"
          placeholder="Search by tracking number, sender, or receiver..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full sm:w-80 px-4 py-2.5 rounded-lg text-sm"
          style={{
            background: "var(--color-bg-input)",
            color: "var(--color-text-primary)",
          }}
        />
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 p-3 rounded-lg text-sm"
             style={{ background: "rgba(239, 68, 68, 0.12)", color: "#f87171", border: "1px solid rgba(239, 68, 68, 0.2)" }}>
          {error}
        </div>
      )}

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="flex flex-col items-center gap-3">
            <div className="spinner" style={{ width: "32px", height: "32px" }} />
            <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Loading shipments...</p>
          </div>
        </div>
      ) : shipments.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <div className="w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center"
               style={{ background: "rgba(99, 102, 241, 0.1)" }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
              <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
              <line x1="12" y1="22.08" x2="12" y2="12" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-1" style={{ color: "var(--color-text-primary)" }}>
            No Shipments Yet
          </h3>
          <p className="text-sm mb-4" style={{ color: "var(--color-text-muted)" }}>
            Create your first shipment to start tracking.
          </p>
          <button
            onClick={() => setShowCreate(true)}
            className="btn-primary inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold text-white border-none cursor-pointer"
          >
            Create First Shipment
          </button>
        </div>
      ) : (
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                  {["Tracking #", "Sender", "Receiver", "Route", "Carrier", "Weight", "Status", "Actions"].map((col) => (
                    <th key={col} className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider"
                        style={{ color: "var(--color-text-muted)" }}>
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {shipments.map((s) => (
                  <tr key={s._id} className="table-row-hover" style={{ borderBottom: "1px solid var(--color-border)" }}>
                    <td className="px-5 py-3.5 font-medium" style={{ color: "var(--color-text-primary)" }}>
                      {s.tracking_number}
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {s.sender_name}
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {s.receiver_name}
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {s.origin} → {s.destination}
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold"
                            style={{
                              background: `${CARRIER_COLORS[s.carrier] || "#94a3b8"}18`,
                              color: CARRIER_COLORS[s.carrier] || "#94a3b8",
                              border: `1px solid ${CARRIER_COLORS[s.carrier] || "#94a3b8"}30`,
                            }}>
                        {s.carrier}
                      </span>
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {s.weight} {s.unit}
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold"
                            style={{
                              background: STATUS_COLORS[s.status]?.bg || "rgba(156,163,175,0.12)",
                              color: STATUS_COLORS[s.status]?.color || "#9ca3af",
                              border: `1px solid ${STATUS_COLORS[s.status]?.border || "rgba(156,163,175,0.25)"}`,
                            }}>
                        {formatStatus(s.status)}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <button
                        onClick={() => setDeleteId(s._id)}
                        className="p-1.5 rounded-lg transition-colors cursor-pointer"
                        style={{ color: "var(--color-text-muted)", background: "none", border: "none" }}
                        title="Delete"
                      >
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                          <polyline points="3 6 5 6 21 6" />
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                        </svg>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Create shipment modal */}
      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="glass-card max-w-lg w-full page-enter flex flex-col my-auto" style={{ maxHeight: "85vh" }}>
            <div className="p-6 pb-3">
              <h3 className="text-lg font-semibold" style={{ color: "var(--color-text-primary)" }}>
                New Shipment
              </h3>
            </div>
            <form onSubmit={handleCreate} className="flex flex-col flex-1 min-h-0">
              <div className="px-6 space-y-3 overflow-y-auto flex-1">
                <input
                  type="text" placeholder="Tracking Number *" required
                  value={form.tracking_number} onChange={(e) => setForm({ ...form, tracking_number: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                />
                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="text" placeholder="Sender Name *" required
                    value={form.sender_name} onChange={(e) => setForm({ ...form, sender_name: e.target.value })}
                    className="w-full px-3 py-2.5 rounded-lg text-sm"
                    style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                  />
                  <input
                    type="text" placeholder="Receiver Name *" required
                    value={form.receiver_name} onChange={(e) => setForm({ ...form, receiver_name: e.target.value })}
                    className="w-full px-3 py-2.5 rounded-lg text-sm"
                    style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                  />
                </div>
                <input
                  type="text" placeholder="Sender Address *" required
                  value={form.sender_address} onChange={(e) => setForm({ ...form, sender_address: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                />
                <input
                  type="text" placeholder="Receiver Address *" required
                  value={form.receiver_address} onChange={(e) => setForm({ ...form, receiver_address: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                />
                <div className="grid grid-cols-2 gap-3">
                  <input
                    type="text" placeholder="Origin *" required
                    value={form.origin} onChange={(e) => setForm({ ...form, origin: e.target.value })}
                    className="w-full px-3 py-2.5 rounded-lg text-sm"
                    style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                  />
                  <input
                    type="text" placeholder="Destination *" required
                    value={form.destination} onChange={(e) => setForm({ ...form, destination: e.target.value })}
                    className="w-full px-3 py-2.5 rounded-lg text-sm"
                    style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                  />
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <select
                    value={form.carrier} onChange={(e) => setForm({ ...form, carrier: e.target.value })}
                    className="w-full px-3 py-2.5 rounded-lg text-sm"
                    style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                  >
                    {CARRIERS.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                  <input
                    type="number" step="0.01" placeholder="Weight *" required
                    value={form.weight} onChange={(e) => setForm({ ...form, weight: e.target.value })}
                    className="w-full px-3 py-2.5 rounded-lg text-sm"
                    style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                  />
                  <select
                    value={form.unit} onChange={(e) => setForm({ ...form, unit: e.target.value })}
                    className="w-full px-3 py-2.5 rounded-lg text-sm"
                    style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                  >
                    <option value="KG">KG</option>
                    <option value="LB">LB</option>
                  </select>
                </div>
                <textarea
                  placeholder="Notes (optional)" rows={2}
                  value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm resize-none"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                />
              </div>
              <div className="flex items-center gap-3 justify-end p-6 pt-4" style={{ borderTop: "1px solid var(--color-border)" }}>
                <button
                  type="button" onClick={() => { setShowCreate(false); setForm({ ...EMPTY_FORM }); }}
                  className="px-4 py-2 rounded-lg text-sm font-medium cursor-pointer"
                  style={{ background: "var(--color-bg-input)", border: "1px solid var(--color-border)", color: "var(--color-text-secondary)" }}
                >
                  Cancel
                </button>
                <button
                  type="submit" disabled={creating}
                  className="btn-primary px-4 py-2 rounded-lg text-sm font-semibold text-white border-none cursor-pointer disabled:opacity-50"
                >
                  {creating ? "Creating..." : "Create Shipment"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteId && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="glass-card p-6 max-w-sm w-full mx-4 page-enter">
            <h3 className="text-lg font-semibold mb-2" style={{ color: "var(--color-text-primary)" }}>
              Delete Shipment?
            </h3>
            <p className="text-sm mb-5" style={{ color: "var(--color-text-secondary)" }}>
              This action cannot be undone. The shipment record will be permanently removed.
            </p>
            <div className="flex items-center gap-3 justify-end">
              <button
                onClick={() => setDeleteId(null)}
                className="px-4 py-2 rounded-lg text-sm font-medium cursor-pointer"
                style={{ background: "var(--color-bg-input)", border: "1px solid var(--color-border)", color: "var(--color-text-secondary)" }}
              >
                Cancel
              </button>
              <button
                onClick={handleDelete} disabled={deleting}
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
