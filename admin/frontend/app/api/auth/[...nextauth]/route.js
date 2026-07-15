/**
 * app/api/auth/[...nextauth]/route.js — NextAuth.js v5 Configuration
 *
 * Configures two authentication providers:
 * 1. CredentialsProvider: Email/password login via Flask /api/auth/login
 * 2. GoogleProvider: OAuth via Google Cloud Console
 *
 * JWT callback stores the Flask JWT token and user role.
 * Session callback exposes accessToken and role to the client.
 */

import NextAuth from "next-auth";
import CredentialsProvider from "next-auth/providers/credentials";
import GoogleProvider from "next-auth/providers/google";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001";

const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    // --- Credentials Provider (email + password via Flask API) ---
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email", placeholder: "admin@example.com" },
        password: { label: "Password", type: "password" },
      },
      async authorize(credentials) {
        try {
          const res = await fetch(`${API_URL}/api/auth/login`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              email: credentials.email,
              password: credentials.password,
            }),
          });

          const data = await res.json();

          if (!res.ok) {
            throw new Error(data.error || "Login failed");
          }

          // Return user object — this gets stored in the JWT
          return {
            id: data.user.id,
            email: data.user.email,
            role: data.user.role,
            accessToken: data.token,
          };
        } catch (error) {
          console.error("Auth error:", error.message);
          return null;
        }
      },
    }),

    // --- Google OAuth Provider ---
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID || "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET || "",
    }),
  ],

  // Use JWT strategy (no database sessions)
  session: {
    strategy: "jwt",
    maxAge: 7 * 24 * 60 * 60, // 7 days
  },

  pages: {
    signIn: "/",        // Custom login page
    error: "/",         // Redirect errors to login
  },

  callbacks: {
    /**
     * JWT callback — runs when JWT is created or updated.
     * Stores the Flask accessToken and user role in the token.
     */
    async jwt({ token, user, account }) {
      if (user) {
        if (account && account.provider === "google") {
          try {
            const res = await fetch(`${API_URL}/api/auth/google`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ email: user.email }),
            });
            const data = await res.json();
            if (res.ok) {
              token.accessToken = data.token;
              token.role = data.user.role;
              token.userId = data.user.id;
            } else {
              console.error("Google auth backend sync failed:", data.error);
            }
          } catch (error) {
            console.error("Google auth backend sync error:", error);
          }
        } else {
          // Credentials login
          token.accessToken = user.accessToken;
          token.role = user.role;
          token.userId = user.id;
        }
      }
      return token;
    },

    /**
     * Session callback — exposes token data to the client session.
     */
    async session({ session, token }) {
      session.accessToken = token.accessToken;
      session.user.role = token.role;
      session.user.id = token.userId;
      return session;
    },
  },

  secret: process.env.NEXTAUTH_SECRET,
});

export const { GET, POST } = handlers;
