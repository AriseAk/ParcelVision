/**
 * app/admin/rates/new/page.js — Create New Rate Plan
 *
 * Uses the shared RateForm component.
 * Submits POST /api/rates to create a new rate plan.
 */

"use client";

import { useState } from "react";
import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import RateForm from "../../../../components/RateForm";
import { apiPost } from "../../../../lib/api";

export default function NewRatePage() {
  const { data: session } = useSession();
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (formData) => {
    try {
      setLoading(true);
      setError("");
      await apiPost("/api/rates", formData, session?.accessToken);
      router.push("/admin/rates");
    } catch (err) {
      setError(err.message || "Failed to create rate plan");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-3xl">
      {/* Page header */}
      <div className="mb-6">
        <div className="flex items-center gap-2 mb-1">
          <a
            href="/admin/rates"
            className="text-sm no-underline flex items-center gap-1"
            style={{ color: "var(--color-text-muted)" }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            Rates
          </a>
          <span style={{ color: "var(--color-text-muted)" }}>/</span>
          <span className="text-sm" style={{ color: "var(--color-text-secondary)" }}>New</span>
        </div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
          Create Rate Plan
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--color-text-secondary)" }}>
          Define a new shipping rate with weight-based pricing tiers.
        </p>
      </div>

      {/* Error message */}
      {error && (
        <div className="mb-4 p-3 rounded-lg text-sm"
             style={{ background: "rgba(239, 68, 68, 0.12)", color: "#f87171", border: "1px solid rgba(239, 68, 68, 0.2)" }}>
          {error}
        </div>
      )}

      {/* Form */}
      <div className="glass-card p-6">
        <RateForm onSubmit={handleSubmit} loading={loading} submitLabel="Create Rate Plan" />
      </div>
    </div>
  );
}
