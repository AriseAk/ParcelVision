/**
 * app/admin/users/page.js — Users Management Page
 *
 * Lists all registered users from the Flask API.
 * Features:
 * - Fetches GET /api/users on mount
 * - Styled data table with role badges
 * - Search by email
 * - Toggle user role (admin/user)
 * - Delete user with confirmation
 * - Loading and empty states
 */

"use client";

import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { apiGet, apiPut, apiDelete } from "../../../lib/api";

const ROLE_COLORS = {
  admin: { bg: "rgba(168, 85, 247, 0.12)", color: "#a78bfa", border: "rgba(168, 85, 247, 0.25)" },
  user:  { bg: "rgba(59, 130, 246, 0.12)", color: "#60a5fa", border: "rgba(59, 130, 246, 0.25)" },
};

const PROVIDER_COLORS = {
  local:  { bg: "rgba(34, 197, 94, 0.12)",  color: "#4ade80", border: "rgba(34, 197, 94, 0.25)" },
  google: { bg: "rgba(245, 158, 11, 0.12)", color: "#f59e0b", border: "rgba(245, 158, 11, 0.25)" },
};

export default function UsersPage() {
  const { data: session } = useSession();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [deleteId, setDeleteId] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // Fetch users
  const fetchUsers = useCallback(async () => {
    if (!session?.accessToken) return;
    try {
      setLoading(true);
      const params = search ? `?search=${encodeURIComponent(search)}` : "";
      const data = await apiGet(`/api/users${params}`, session.accessToken);
      setUsers(data.users || []);
      setError("");
    } catch (err) {
      setError(err.message || "Failed to load users");
    } finally {
      setLoading(false);
    }
  }, [session?.accessToken, search]);

  useEffect(() => { fetchUsers(); }, [fetchUsers]);

  // Toggle role
  const handleToggleRole = async (userId, currentRole) => {
    const newRole = currentRole === "admin" ? "user" : "admin";
    try {
      await apiPut(`/api/users/${userId}`, { role: newRole }, session.accessToken);
      setUsers((prev) =>
        prev.map((u) => (u._id === userId ? { ...u, role: newRole } : u))
      );
    } catch (err) {
      setError(err.message || "Failed to update user role");
    }
  };

  // Delete user
  const handleDelete = async () => {
    if (!deleteId) return;
    try {
      setDeleting(true);
      await apiDelete(`/api/users/${deleteId}`, session.accessToken);
      setUsers((prev) => prev.filter((u) => u._id !== deleteId));
      setDeleteId(null);
    } catch (err) {
      setError(err.message || "Failed to delete user");
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

  return (
    <div>
      {/* Page header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
          Users
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--color-text-secondary)" }}>
          View and manage all registered user accounts.
        </p>
      </div>

      {/* Search bar */}
      <div className="mb-5">
        <input
          id="user-search"
          type="text"
          placeholder="Search by email..."
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
            <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Loading users...</p>
          </div>
        </div>
      ) : users.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <div className="w-16 h-16 rounded-full mx-auto mb-4 flex items-center justify-center"
               style={{ background: "rgba(99, 102, 241, 0.1)" }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold mb-1" style={{ color: "var(--color-text-primary)" }}>
            No Users Found
          </h3>
          <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
            No registered users match your search criteria.
          </p>
        </div>
      ) : (
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr style={{ borderBottom: "1px solid var(--color-border)" }}>
                  {["Email", "Role", "Auth Provider", "Joined", "Actions"].map((col) => (
                    <th key={col} className="px-5 py-3.5 text-left text-xs font-semibold uppercase tracking-wider"
                        style={{ color: "var(--color-text-muted)" }}>
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {users.map((user) => {
                  const isSelf = user._id === session?.user?.id;
                  return (
                    <tr key={user._id} className="table-row-hover" style={{ borderBottom: "1px solid var(--color-border)" }}>
                      <td className="px-5 py-3.5 font-medium" style={{ color: "var(--color-text-primary)" }}>
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                               style={{ background: "linear-gradient(135deg, var(--color-accent), var(--color-accent-hover))" }}>
                            {user.email?.[0]?.toUpperCase() || "?"}
                          </div>
                          <span>{user.email}</span>
                          {isSelf && (
                            <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: "rgba(99,102,241,0.12)", color: "#818cf8" }}>
                              You
                            </span>
                          )}
                        </div>
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold"
                              style={{
                                background: ROLE_COLORS[user.role]?.bg || ROLE_COLORS.user.bg,
                                color: ROLE_COLORS[user.role]?.color || ROLE_COLORS.user.color,
                                border: `1px solid ${ROLE_COLORS[user.role]?.border || ROLE_COLORS.user.border}`,
                              }}>
                          {user.role?.charAt(0).toUpperCase() + user.role?.slice(1)}
                        </span>
                      </td>
                      <td className="px-5 py-3.5">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold"
                              style={{
                                background: PROVIDER_COLORS[user.auth_provider]?.bg || PROVIDER_COLORS.local.bg,
                                color: PROVIDER_COLORS[user.auth_provider]?.color || PROVIDER_COLORS.local.color,
                                border: `1px solid ${PROVIDER_COLORS[user.auth_provider]?.border || PROVIDER_COLORS.local.border}`,
                              }}>
                          {user.auth_provider === "google" ? "Google" : "Local"}
                        </span>
                      </td>
                      <td className="px-5 py-3.5" style={{ color: "var(--color-text-secondary)" }}>
                        {formatDate(user.created_at)}
                      </td>
                      <td className="px-5 py-3.5">
                        <div className="flex items-center gap-2">
                          {!isSelf && (
                            <>
                              <button
                                onClick={() => handleToggleRole(user._id, user.role)}
                                className="p-1.5 rounded-lg transition-colors cursor-pointer"
                                style={{ color: "var(--color-text-muted)", background: "none", border: "none" }}
                                title={`Make ${user.role === "admin" ? "User" : "Admin"}`}
                              >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                                </svg>
                              </button>
                              <button
                                onClick={() => setDeleteId(user._id)}
                                className="p-1.5 rounded-lg transition-colors cursor-pointer"
                                style={{ color: "var(--color-text-muted)", background: "none", border: "none" }}
                                title="Delete"
                              >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                                  <polyline points="3 6 5 6 21 6" />
                                  <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                                </svg>
                              </button>
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
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
              Delete User?
            </h3>
            <p className="text-sm mb-5" style={{ color: "var(--color-text-secondary)" }}>
              This action cannot be undone. The user account will be permanently removed.
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
