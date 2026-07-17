/**
 * app/admin/transactions/page.js — Transactions Management Page
 *
 * Lists all transactions from the Flask API.
 * Features:
 * - Fetches GET /api/transactions on mount
 * - Styled data table with status and payment method badges
 * - Search by shipment ID or description
 * - Create new transaction modal
 * - Delete confirmation modal
 * - Loading and empty states
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { apiGet, apiPost, apiDelete } from "../../../lib/api";

const CURRENCIES = ["USD", "EUR", "GBP", "INR", "AED"];
const PAYMENT_METHODS = [
  { value: "credit_card",   label: "Credit Card" },
  { value: "debit_card",    label: "Debit Card" },
  { value: "paypal",        label: "PayPal" },
  { value: "bank_transfer", label: "Bank Transfer" },
  { value: "cash",          label: "Cash" },
  { value: "upi",           label: "UPI" },
];
const TXN_STATUSES = ["pending", "completed", "failed", "refunded"];

const STATUS_COLORS = {
  pending:   { bg: "rgba(245, 158, 11, 0.12)", color: "#f59e0b", border: "rgba(245, 158, 11, 0.25)" },
  completed: { bg: "rgba(34, 197, 94, 0.12)",  color: "#4ade80", border: "rgba(34, 197, 94, 0.25)" },
  failed:    { bg: "rgba(239, 68, 68, 0.12)",  color: "#f87171", border: "rgba(239, 68, 68, 0.25)" },
  refunded:  { bg: "rgba(168, 85, 247, 0.12)", color: "#a78bfa", border: "rgba(168, 85, 247, 0.25)" },
};

const METHOD_COLORS = {
  credit_card:   "#818cf8",
  debit_card:    "#60a5fa",
  paypal:        "#fbbf24",
  bank_transfer: "#4ade80",
  cash:          "#9ca3af",
  upi:           "#f472b6",
};

const CURRENCY_SYMBOLS = { USD: "$", EUR: "€", GBP: "£", INR: "₹", AED: "د.إ" };

const EMPTY_FORM = {
  shipment_id: "", amount: "", currency: "USD",
  payment_method: "credit_card", status: "pending", description: "",
};

export default function TransactionsPage() {
  const { data: session } = useSession();
  const [transactions, setTransactions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [deleteId, setDeleteId] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [creating, setCreating] = useState(false);

  // Fetch transactions
  const fetchTransactions = useCallback(async () => {
    if (!session?.accessToken) return;
    try {
      setLoading(true);
      const params = search ? `?search=${encodeURIComponent(search)}` : "";
      const data = await apiGet(`/api/transactions${params}`, session.accessToken);
      setTransactions(data.transactions || []);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load transactions");
    } finally {
      setLoading(false);
    }
  }, [session?.accessToken, search]);

  useEffect(() => { fetchTransactions(); }, [fetchTransactions]);

  // Create transaction
  const handleCreate = async (e) => {
    e.preventDefault();
    try {
      setCreating(true);
      await apiPost("/api/transactions", form, session.accessToken);
      setShowCreate(false);
      setForm({ ...EMPTY_FORM });
      fetchTransactions();
    } catch (err) {
      setError(err.message || "Failed to create transaction");
    } finally {
      setCreating(false);
    }
  };

  // Delete transaction
  const handleDelete = async () => {
    if (!deleteId) return;
    try {
      setDeleting(true);
      await apiDelete(`/api/transactions/${deleteId}`, session.accessToken);
      setTransactions((prev) => prev.filter((t) => t._id !== deleteId));
      setDeleteId(null);
    } catch (err) {
      setError(err.message || "Failed to delete transaction");
    } finally {
      setDeleting(false);
    }
  };

  const formatDate = (dateStr) => {
    if (!dateStr) return "—";
    try {
      return new Date(dateStr).toLocaleDateString("en-US", {
        year: "numeric", month: "short", day: "numeric",
      });
    } catch {
      return "—";
    }
  };

  const formatAmount = (amount, currency) => {
    const symbol = CURRENCY_SYMBOLS[currency] || currency;
    return `${symbol}${Number(amount).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatMethod = (method) => {
    const found = PAYMENT_METHODS.find((m) => m.value === method);
    return found ? found.label : method;
  };

  return (
    <div>
      {/* Page header */}
      <div className="flex items-center justify-between mb-6 flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
            Transactions
          </h1>
          <p className="mt-1 text-sm" style={{ color: "var(--color-text-secondary)" }}>
            Monitor payments and financial records for shipments.
          </p>
        </div>

        <button
          id="create-transaction-btn"
          onClick={() => setShowCreate(true)}
          className="btn-primary px-5 py-2.5 rounded-lg text-sm font-semibold text-white border-none cursor-pointer flex items-center gap-2"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19" />
            <line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          New Transaction
        </button>
      </div>

      {/* Search bar */}
      <div className="mb-5">
        <input
          id="transaction-search"
          type="text"
          placeholder="Search by shipment ID or description..."
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
            <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Loading transactions...</p>
          </div>
        </div>
      ) : transactions.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <div className="w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center"
               style={{ background: "rgba(99, 102, 241, 0.1)" }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <rect x="1" y="4" width="22" height="16" rx="2" ry="2" />
              <line x1="1" y1="10" x2="23" y2="10" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-1" style={{ color: "var(--color-text-primary)" }}>
            No Transactions Yet
          </h3>
          <p className="text-sm mb-4" style={{ color: "var(--color-text-muted)" }}>
            Record your first transaction to start tracking payments.
          </p>
          <button
            onClick={() => setShowCreate(true)}
            className="btn-primary inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold text-white border-none cursor-pointer"
          >
            Create First Transaction
          </button>
        </div>
      ) : (
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                  {["Shipment ID", "Amount", "Payment Method", "Status", "Description", "Date", "Actions"].map((col) => (
                    <th key={col} className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider"
                        style={{ color: "var(--color-text-muted)" }}>
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {transactions.map((txn) => (
                  <tr key={txn._id} className="table-row-hover" style={{ borderBottom: "1px solid var(--color-border)" }}>
                    <td className="px-5 py-3.5 font-medium font-mono text-xs" style={{ color: "var(--color-text-primary)" }}>
                      {txn.shipment_id?.substring(0, 12)}...
                    </td>
                    <td className="px-5 py-3.5 font-semibold" style={{ color: "var(--color-text-primary)" }}>
                      {formatAmount(txn.amount, txn.currency)}
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold"
                            style={{
                              background: `${METHOD_COLORS[txn.payment_method] || "#94a3b8"}18`,
                              color: METHOD_COLORS[txn.payment_method] || "#94a3b8",
                              border: `1px solid ${METHOD_COLORS[txn.payment_method] || "#94a3b8"}30`,
                            }}>
                        {formatMethod(txn.payment_method)}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold"
                            style={{
                              background: STATUS_COLORS[txn.status]?.bg || "rgba(156,163,175,0.12)",
                              color: STATUS_COLORS[txn.status]?.color || "#9ca3af",
                              border: `1px solid ${STATUS_COLORS[txn.status]?.border || "rgba(156,163,175,0.25)"}`,
                            }}>
                        {txn.status?.charAt(0).toUpperCase() + txn.status?.slice(1)}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 max-w-[200px] truncate" style={{ color: "var(--color-text-secondary)" }}>
                      {txn.description || "—"}
                    </td>
                    <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                      {formatDate(txn.created_at)}
                    </td>
                    <td className="px-5 py-3.5">
                      <button
                        onClick={() => setDeleteId(txn._id)}
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

      {/* Create transaction modal */}
      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="glass-card p-6 max-w-lg w-full mx-4 page-enter">
            <h3 className="text-lg font-semibold mb-4" style={{ color: "var(--color-text-primary)" }}>
              New Transaction
            </h3>
            <form onSubmit={handleCreate} className="space-y-3">
              <input
                type="text" placeholder="Shipment ID *" required
                value={form.shipment_id} onChange={(e) => setForm({ ...form, shipment_id: e.target.value })}
                className="w-full px-3 py-2.5 rounded-lg text-sm"
                style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
              />
              <div className="grid grid-cols-2 gap-3">
                <input
                  type="number" step="0.01" placeholder="Amount *" required
                  value={form.amount} onChange={(e) => setForm({ ...form, amount: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                />
                <select
                  value={form.currency} onChange={(e) => setForm({ ...form, currency: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                >
                  {CURRENCIES.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <select
                  value={form.payment_method} onChange={(e) => setForm({ ...form, payment_method: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                >
                  {PAYMENT_METHODS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
                </select>
                <select
                  value={form.status} onChange={(e) => setForm({ ...form, status: e.target.value })}
                  className="w-full px-3 py-2.5 rounded-lg text-sm"
                  style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
                >
                  {TXN_STATUSES.map((s) => (
                    <option key={s} value={s}>{s.charAt(0).toUpperCase() + s.slice(1)}</option>
                  ))}
                </select>
              </div>
              <textarea
                placeholder="Description (optional)" rows={2}
                value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })}
                className="w-full px-3 py-2.5 rounded-lg text-sm resize-none"
                style={{ background: "var(--color-bg-input)", color: "var(--color-text-primary)" }}
              />
              <div className="flex items-center gap-3 justify-end pt-2">
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
                  {creating ? "Creating..." : "Create Transaction"}
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
              Delete Transaction?
            </h3>
            <p className="text-sm mb-5" style={{ color: "var(--color-text-secondary)" }}>
              This action cannot be undone. The transaction record will be permanently removed.
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
