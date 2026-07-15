/**
 * app/admin/layout.js — Admin Layout
 *
 * Protected layout wrapper for all /admin/* pages:
 * - Checks authentication (redirects to login if unauthenticated)
 * - Renders sidebar + main content area
 * - Loading state while checking session
 */

"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import Sidebar from "../../components/Sidebar";

export default function AdminLayout({ children }) {
  const { data: session, status } = useSession();
  const router = useRouter();

  // Redirect to login if not authenticated
  useEffect(() => {
    if (status === "unauthenticated") {
      router.push("/");
    }
  }, [status, router]);

  // Show loading spinner while checking auth
  if (status === "loading") {
    return (
      <div className="min-h-screen flex items-center justify-center"
           style={{ background: "var(--color-bg-primary)" }}>
        <div className="flex flex-col items-center gap-3">
          <div className="spinner" style={{ width: "32px", height: "32px" }} />
          <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Loading...</p>
        </div>
      </div>
    );
  }

  // Don't render content if unauthenticated
  if (status === "unauthenticated") {
    return null;
  }

  return (
    <div className="min-h-screen" style={{ background: "var(--color-bg-primary)" }}>
      {/* Sidebar */}
      <Sidebar />

      {/* Main content area — offset by sidebar width on desktop */}
      <main className="lg:ml-64 min-h-screen">
        <div className="p-6 lg:p-8 page-enter">
          {children}
        </div>
      </main>
    </div>
  );
}
