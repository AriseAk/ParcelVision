/**
 * app/admin/rates/edit/[id]/page.js — Edit Rate Plan
 *
 * Fetches an existing rate plan by ID, pre-populates the RateForm,
 * and submits PUT /api/rates/<id> to update it.
 */

"use client";

import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { useRouter, useParams } from "next/navigation";
import RateForm from "../../../../../components/RateForm";
import { apiGet, apiPut } from "../../../../../lib/api";

export default function EditRatePage() {
  const { data: session } = useSession();
  const router = useRouter();
  const params = useParams();
  const rateId = params.id;

  const [initialData, setInitialData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [fetching, setFetching] = useState(true);
  const [error, setError] = useState("");

  // Fetch existing rate plan
  useEffect(() => {
    const fetchRate = async () => {
      if (!session?.accessToken || !rateId) return;

      try {
        setFetching(true);
        const data = await apiGet(`/api/rates/${rateId}`, session.accessToken);
        setInitialData(data.rate);
      } catch (err) {
        setError(err.message || "Failed to load rate plan");
      } finally {
        setFetching(false);
      }
    };

    fetchRate();
  }, [session?.accessToken, rateId]);

  // Handle form submission
  const handleSubmit = async (formData) => {
    try {
      setLoading(true);
      setError("");
      await apiPut(`/api/rates/${rateId}`, formData, session?.accessToken);
      router.push("/admin/rates");
    } catch (err) {
      setError(err.message || "Failed to update rate plan");
    } finally {
      setLoading(false);
    }
  };

  // Loading state while fetching existing data
  if (fetching) {
    return (
      <div className="flex items-center justify-center py-20">
        <div className="flex flex-col items-center gap-3">
          <div className="spinner" style={{ width: "32px", height: "32px" }} />
          <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Loading rate plan...</p>
        </div>
      </div>
    );
  }

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
          <span className="text-sm" style={{ color: "var(--color-text-secondary)" }}>Edit</span>
        </div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
          Edit Rate Plan
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--color-text-secondary)" }}>
          Update the shipping rate details and weight tiers.
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
        <RateForm
          initialData={initialData}
          onSubmit={handleSubmit}
          loading={loading}
          submitLabel="Update Rate Plan"
        />
      </div>
    </div>
  );
}
