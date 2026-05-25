export const metadata = {
  title: "TRIPOD — Monocular Box Metrology",
  description: "Measure 3D box dimensions with a single smartphone camera using the Human Tripod constraint.",
  viewport: "width=device-width, initial-scale=1, viewport-fit=cover",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, background: "#000", overflow: "hidden" }}>
        {children}
      </body>
    </html>
  );
}
