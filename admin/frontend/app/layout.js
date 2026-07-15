/**
 * app/layout.js — Root Layout
 *
 * Wraps the entire admin app with:
 * - Inter font from Google Fonts
 * - SessionProvider for NextAuth
 * - Global CSS
 */

import "./globals.css";
import Providers from "./providers";

export const metadata = {
  title: "ParcelVision Admin | Logistics Management",
  description: "Admin panel for managing international shipping rates, shipments, and users.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        {/* Google Fonts — Inter */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
