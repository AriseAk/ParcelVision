/**
 * components/RateForm.js — Reusable Rate Plan Form
 *
 * Shared form component used by both Create and Edit pages.
 * Features:
 * - Dropdowns: Origin, Destination, Carrier, Category, Currency, Unit
 * - Text input: Estimated Days
 * - Checkbox: Food Allowed
 * - Dynamic Weight Tiers: Array of {start_weight, end_weight, price}
 *   with "Add Tier" and "Remove" buttons
 * - Full form state managed via useState
 * - Submits entire form as a single JSON payload
 *
 * Props:
 * - initialData: Pre-filled form data (for editing)
 * - onSubmit: Callback(formData) when form is submitted
 * - loading: Boolean to show loading state on submit button
 * - submitLabel: Text for submit button ("Create" or "Update")
 */

"use client";

import { useState, useEffect } from "react";

// ---- Dropdown options ----
const CARRIERS = ["DHL", "FedEx", "UPS", "Aramex"];
const CATEGORIES = ["Medicine", "Documents", "General", "Electronics", "Food"];
const CURRENCIES = ["USD", "EUR", "GBP", "INR", "AED"];
const UNITS = ["KG", "LB"];

// Popular countries for origin/destination
const COUNTRIES = [
  "United States", "United Kingdom", "India", "United Arab Emirates",
  "Saudi Arabia", "Canada", "Germany", "France", "Australia", "China",
  "Japan", "South Korea", "Singapore", "Malaysia", "Thailand",
  "Qatar", "Kuwait", "Bahrain", "Oman", "Egypt", "Turkey",
  "Netherlands", "Belgium", "Italy", "Spain", "Brazil", "Mexico",
  "South Africa", "Nigeria", "Kenya", "Pakistan", "Bangladesh",
  "Sri Lanka", "Philippines", "Indonesia", "Vietnam",
];

// Default empty tier
const EMPTY_TIER = { start_weight: "", end_weight: "", price: "" };

// Default form state
const DEFAULT_FORM = {
  origin: "",
  destination: "",
  carrier: "",
  category: "",
  currency: "USD",
  unit: "KG",
  estimated_days: "",
  food_allowed: false,
  tiers: [{ ...EMPTY_TIER }],
};

export default function RateForm({ initialData, onSubmit, loading, submitLabel = "Save" }) {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [errors, setErrors] = useState({});

  // Pre-fill form with initial data when editing
  useEffect(() => {
    if (initialData) {
      setForm({
        origin: initialData.origin || "",
        destination: initialData.destination || "",
        carrier: initialData.carrier || "",
        category: initialData.category || "",
        currency: initialData.currency || "USD",
        unit: initialData.unit || "KG",
        estimated_days: initialData.estimated_days?.toString() || "",
        food_allowed: initialData.food_allowed || false,
        tiers:
          initialData.tiers?.length > 0
            ? initialData.tiers.map((t) => ({
                start_weight: t.start_weight?.toString() || "",
                end_weight: t.end_weight?.toString() || "",
                price: t.price?.toString() || "",
              }))
            : [{ ...EMPTY_TIER }],
      });
    }
  }, [initialData]);

  // Update a field
  const handleChange = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
    // Clear error for this field
    if (errors[field]) {
      setErrors((prev) => ({ ...prev, [field]: null }));
    }
  };

  // ---- Tier management ----
  const handleTierChange = (index, field, value) => {
    setForm((prev) => {
      const newTiers = [...prev.tiers];
      newTiers[index] = { ...newTiers[index], [field]: value };
      return { ...prev, tiers: newTiers };
    });
  };

  const addTier = () => {
    setForm((prev) => ({ ...prev, tiers: [...prev.tiers, { ...EMPTY_TIER }] }));
  };

  const removeTier = (index) => {
    setForm((prev) => ({
      ...prev,
      tiers: prev.tiers.filter((_, i) => i !== index),
    }));
  };

  // ---- Validation ----
  const validate = () => {
    const errs = {};
    if (!form.origin) errs.origin = "Origin is required";
    if (!form.destination) errs.destination = "Destination is required";
    if (!form.carrier) errs.carrier = "Carrier is required";
    if (!form.category) errs.category = "Category is required";
    if (!form.estimated_days || isNaN(form.estimated_days))
      errs.estimated_days = "Valid number required";

    if (form.tiers.length === 0) {
      errs.tiers = "At least one weight tier is required";
    } else {
      form.tiers.forEach((tier, i) => {
        if (!tier.start_weight || isNaN(tier.start_weight))
          errs[`tier_${i}_start`] = "Required";
        if (!tier.end_weight || isNaN(tier.end_weight))
          errs[`tier_${i}_end`] = "Required";
        if (!tier.price || isNaN(tier.price))
          errs[`tier_${i}_price`] = "Required";
      });
    }

    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  // ---- Submit ----
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validate()) return;

    // Build the JSON payload
    const payload = {
      origin: form.origin,
      destination: form.destination,
      carrier: form.carrier,
      category: form.category,
      currency: form.currency,
      unit: form.unit,
      estimated_days: parseInt(form.estimated_days, 10),
      food_allowed: form.food_allowed,
      tiers: form.tiers.map((t) => ({
        start_weight: parseFloat(t.start_weight),
        end_weight: parseFloat(t.end_weight),
        price: parseFloat(t.price),
      })),
    };

    onSubmit(payload);
  };

  // ---- Shared input styles ----
  const inputStyle = {
    background: "var(--color-bg-card)",
    border: "1px solid rgba(99, 102, 241, 0.25)",
    color: "var(--color-text-primary)",
  };

  const errorInputStyle = {
    ...inputStyle,
    border: "1px solid #ef4444",
  };

  const labelClass = "block text-sm font-medium mb-1.5";

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* ---- Origin & Destination ---- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Origin */}
        <div>
          <label className={labelClass} style={{ color: "var(--color-text-secondary)" }}>
            Origin Country <span style={{ color: "#ef4444" }}>*</span>
          </label>
          <select
            id="rate-origin"
            value={form.origin}
            onChange={(e) => handleChange("origin", e.target.value)}
            className="w-full px-4 py-2.5 rounded-lg text-sm input-glow outline-none"
            style={errors.origin ? errorInputStyle : inputStyle}
          >
            <option value="">Select origin...</option>
            {COUNTRIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          {errors.origin && <p className="text-xs mt-1" style={{ color: "#f87171" }}>{errors.origin}</p>}
        </div>

        {/* Destination */}
        <div>
          <label className={labelClass} style={{ color: "var(--color-text-secondary)" }}>
            Destination Country <span style={{ color: "#ef4444" }}>*</span>
          </label>
          <select
            id="rate-destination"
            value={form.destination}
            onChange={(e) => handleChange("destination", e.target.value)}
            className="w-full px-4 py-2.5 rounded-lg text-sm input-glow outline-none"
            style={errors.destination ? errorInputStyle : inputStyle}
          >
            <option value="">Select destination...</option>
            {COUNTRIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          {errors.destination && <p className="text-xs mt-1" style={{ color: "#f87171" }}>{errors.destination}</p>}
        </div>
      </div>

      {/* ---- Carrier & Category ---- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Carrier */}
        <div>
          <label className={labelClass} style={{ color: "var(--color-text-secondary)" }}>
            Carrier <span style={{ color: "#ef4444" }}>*</span>
          </label>
          <select
            id="rate-carrier"
            value={form.carrier}
            onChange={(e) => handleChange("carrier", e.target.value)}
            className="w-full px-4 py-2.5 rounded-lg text-sm input-glow outline-none"
            style={errors.carrier ? errorInputStyle : inputStyle}
          >
            <option value="">Select carrier...</option>
            {CARRIERS.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          {errors.carrier && <p className="text-xs mt-1" style={{ color: "#f87171" }}>{errors.carrier}</p>}
        </div>

        {/* Category */}
        <div>
          <label className={labelClass} style={{ color: "var(--color-text-secondary)" }}>
            Item Category <span style={{ color: "#ef4444" }}>*</span>
          </label>
          <select
            id="rate-category"
            value={form.category}
            onChange={(e) => handleChange("category", e.target.value)}
            className="w-full px-4 py-2.5 rounded-lg text-sm input-glow outline-none"
            style={errors.category ? errorInputStyle : inputStyle}
          >
            <option value="">Select category...</option>
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          {errors.category && <p className="text-xs mt-1" style={{ color: "#f87171" }}>{errors.category}</p>}
        </div>
      </div>

      {/* ---- Currency, Unit, Estimated Days ---- */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Currency */}
        <div>
          <label className={labelClass} style={{ color: "var(--color-text-secondary)" }}>Currency</label>
          <select
            id="rate-currency"
            value={form.currency}
            onChange={(e) => handleChange("currency", e.target.value)}
            className="w-full px-4 py-2.5 rounded-lg text-sm input-glow outline-none"
            style={inputStyle}
          >
            {CURRENCIES.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
        </div>

        {/* Unit */}
        <div>
          <label className={labelClass} style={{ color: "var(--color-text-secondary)" }}>Weight Unit</label>
          <select
            id="rate-unit"
            value={form.unit}
            onChange={(e) => handleChange("unit", e.target.value)}
            className="w-full px-4 py-2.5 rounded-lg text-sm input-glow outline-none"
            style={inputStyle}
          >
            {UNITS.map((u) => (
              <option key={u} value={u}>{u}</option>
            ))}
          </select>
        </div>

        {/* Estimated Days */}
        <div>
          <label className={labelClass} style={{ color: "var(--color-text-secondary)" }}>
            Estimated Days <span style={{ color: "#ef4444" }}>*</span>
          </label>
          <input
            id="rate-estimated-days"
            type="number"
            min="1"
            value={form.estimated_days}
            onChange={(e) => handleChange("estimated_days", e.target.value)}
            placeholder="e.g. 5"
            className="w-full px-4 py-2.5 rounded-lg text-sm input-glow outline-none"
            style={errors.estimated_days ? errorInputStyle : inputStyle}
          />
          {errors.estimated_days && <p className="text-xs mt-1" style={{ color: "#f87171" }}>{errors.estimated_days}</p>}
        </div>
      </div>

      {/* ---- Food Allowed Checkbox ---- */}
      <div className="flex items-center gap-3">
        <input
          id="rate-food-allowed"
          type="checkbox"
          checked={form.food_allowed}
          onChange={(e) => handleChange("food_allowed", e.target.checked)}
          className="w-4 h-4 rounded cursor-pointer accent-indigo-500"
        />
        <label htmlFor="rate-food-allowed" className="text-sm font-medium cursor-pointer" style={{ color: "var(--color-text-secondary)" }}>
          Food Allowed
        </label>
      </div>

      {/* ---- Weight Tiers (Dynamic Array) ---- */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <label className="text-sm font-semibold" style={{ color: "var(--color-text-primary)" }}>
            Weight Tiers <span style={{ color: "#ef4444" }}>*</span>
          </label>
          <button
            id="add-tier-btn"
            type="button"
            onClick={addTier}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors cursor-pointer"
            style={{
              background: "rgba(99, 102, 241, 0.12)",
              color: "#818cf8",
              border: "1px solid rgba(99, 102, 241, 0.25)",
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            Add Tier
          </button>
        </div>

        {errors.tiers && <p className="text-xs mb-2" style={{ color: "#f87171" }}>{errors.tiers}</p>}

        <div className="space-y-3">
          {form.tiers.map((tier, index) => (
            <div
              key={index}
              className="glass-card p-4 flex items-end gap-3 flex-wrap"
              style={{ background: "var(--color-bg-input)" }}
            >
              {/* Tier number badge */}
              <div className="flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold flex-shrink-0 mb-1"
                   style={{ background: "rgba(99, 102, 241, 0.15)", color: "#818cf8" }}>
                {index + 1}
              </div>

              {/* Start Weight */}
              <div className="flex-1 min-w-[120px]">
                <label className="block text-xs mb-1" style={{ color: "var(--color-text-muted)" }}>
                  Start Weight ({form.unit})
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={tier.start_weight}
                  onChange={(e) => handleTierChange(index, "start_weight", e.target.value)}
                  placeholder="0"
                  className="w-full px-3 py-2 rounded-lg text-sm input-glow outline-none"
                  style={errors[`tier_${index}_start`] ? errorInputStyle : inputStyle}
                />
              </div>

              {/* End Weight */}
              <div className="flex-1 min-w-[120px]">
                <label className="block text-xs mb-1" style={{ color: "var(--color-text-muted)" }}>
                  End Weight ({form.unit})
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={tier.end_weight}
                  onChange={(e) => handleTierChange(index, "end_weight", e.target.value)}
                  placeholder="5"
                  className="w-full px-3 py-2 rounded-lg text-sm input-glow outline-none"
                  style={errors[`tier_${index}_end`] ? errorInputStyle : inputStyle}
                />
              </div>

              {/* Price */}
              <div className="flex-1 min-w-[120px]">
                <label className="block text-xs mb-1" style={{ color: "var(--color-text-muted)" }}>
                  Price ({form.currency})
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  value={tier.price}
                  onChange={(e) => handleTierChange(index, "price", e.target.value)}
                  placeholder="25.00"
                  className="w-full px-3 py-2 rounded-lg text-sm input-glow outline-none"
                  style={errors[`tier_${index}_price`] ? errorInputStyle : inputStyle}
                />
              </div>

              {/* Remove button (only if more than 1 tier) */}
              {form.tiers.length > 1 && (
                <button
                  type="button"
                  onClick={() => removeTier(index)}
                  className="p-2 rounded-lg transition-colors cursor-pointer mb-0.5 flex-shrink-0"
                  style={{
                    background: "rgba(239, 68, 68, 0.1)",
                    color: "#f87171",
                    border: "1px solid rgba(239, 68, 68, 0.2)",
                  }}
                  title="Remove tier"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ---- Submit ---- */}
      <div className="flex items-center gap-3 pt-4" style={{ borderTop: "1px solid var(--color-border)" }}>
        <button
          id="rate-submit-btn"
          type="submit"
          disabled={loading}
          className="btn-primary px-6 py-2.5 rounded-lg text-sm font-semibold text-white disabled:opacity-50 disabled:cursor-not-allowed border-none cursor-pointer"
        >
          {loading ? (
            <span className="flex items-center gap-2">
              <span className="spinner" style={{ width: "16px", height: "16px", borderWidth: "2px" }} />
              Saving...
            </span>
          ) : (
            submitLabel
          )}
        </button>

        <a
          href="/admin/rates"
          className="px-5 py-2.5 rounded-lg text-sm font-medium no-underline"
          style={{
            background: "var(--color-bg-input)",
            border: "1px solid var(--color-border)",
            color: "var(--color-text-secondary)",
          }}
        >
          Cancel
        </a>
      </div>
    </form>
  );
}
