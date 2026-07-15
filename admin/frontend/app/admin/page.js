/**
 * app/admin/page.js — Dashboard Page
 *
 * Admin dashboard with summary stat cards.
 * Placeholder data (to be wired to actual API endpoints later).
 */

"use client";

import { useSession } from "next-auth/react";

// Stat cards data
const STATS = [
  {
    label: "Total Shipments",
    value: "—",
    change: "Pending integration",
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
        <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
        <line x1="12" y1="22.08" x2="12" y2="12" />
      </svg>
    ),
    color: "#6366f1",
  },
  {
    label: "Active Rate Plans",
    value: "—",
    change: "Pending integration",
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <line x1="12" y1="1" x2="12" y2="23" />
        <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
      </svg>
    ),
    color: "#22c55e",
  },
  {
    label: "Registered Users",
    value: "—",
    change: "Pending integration",
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
      </svg>
    ),
    color: "#f59e0b",
  },
  {
    label: "Revenue",
    value: "—",
    change: "Pending integration",
    icon: (
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
        <polyline points="17 6 23 6 23 12" />
      </svg>
    ),
    color: "#8b5cf6",
  },
];

export default function DashboardPage() {
  const { data: session } = useSession();

  return (
    <div>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
          Dashboard
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--color-text-secondary)" }}>
          Welcome back{session?.user?.email ? `, ${session.user.email}` : ""}. Here&apos;s your logistics overview.
        </p>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
        {STATS.map((stat, i) => (
          <div
            key={i}
            className="glass-card p-5 flex items-start justify-between group hover:scale-[1.02] transition-transform duration-200"
          >
            <div>
              <p className="text-xs font-medium uppercase tracking-wider mb-1" style={{ color: "var(--color-text-muted)" }}>
                {stat.label}
              </p>
              <p className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
                {stat.value}
              </p>
              <p className="text-xs mt-1" style={{ color: "var(--color-text-muted)" }}>
                {stat.change}
              </p>
            </div>
            <div
              className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
              style={{ background: `${stat.color}15`, color: stat.color }}
            >
              {stat.icon}
            </div>
          </div>
        ))}
      </div>

      {/* Quick actions section */}
      <div className="mt-8">
        <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--color-text-primary)" }}>
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <a
            href="/admin/rates/new"
            className="glass-card p-5 flex items-center gap-4 no-underline group hover:scale-[1.01] transition-transform duration-200"
            style={{ cursor: "pointer" }}
          >
            <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                 style={{ background: "rgba(99, 102, 241, 0.12)", color: "#818cf8" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>New Rate Plan</p>
              <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>Create a shipping rate</p>
            </div>
          </a>

          <a
            href="/admin/rates"
            className="glass-card p-5 flex items-center gap-4 no-underline group hover:scale-[1.01] transition-transform duration-200"
            style={{ cursor: "pointer" }}
          >
            <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                 style={{ background: "rgba(34, 197, 94, 0.12)", color: "#4ade80" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>View All Rates</p>
              <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>Manage existing plans</p>
            </div>
          </a>

          <div className="glass-card p-5 flex items-center gap-4 opacity-50">
            <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                 style={{ background: "rgba(245, 158, 11, 0.12)", color: "#fbbf24" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>Settings</p>
              <p className="text-xs" style={{ color: "var(--color-text-muted)" }}>Coming soon</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
