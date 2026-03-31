// frontend/app/layout.js
export const metadata = {
  title: "ParcelVision | AI Logistics",
  description: "Instant volumetric dimensioning via AR.",
  viewport: {
    width: "device-width",
    initialScale: 1,
    maximumScale: 1,
    userScalable: false, // Prevents pinch-to-zoom on mobile
  },
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0, backgroundColor: "#0f0f14", overscrollBehavior: "none" }}>
        {children}
      </body>
    </html>
  );
}