/**
 * app/providers.js — Client-side session provider wrapper
 *
 * Wraps the app with NextAuth's SessionProvider so that
 * useSession() works in all client components.
 */

"use client";

import { SessionProvider } from "next-auth/react";

export default function Providers({ children }) {
  return <SessionProvider>{children}</SessionProvider>;
}
