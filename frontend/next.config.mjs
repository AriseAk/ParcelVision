/** @type {import('next').NextConfig} */
const nextConfig = {
  /* config options here */
  reactCompiler: true,
  allowedDevOrigins: [
    "http://192.168.1.103:3000",
    "https://*.ngrok-free.app",
      "https://*.ngrok-free.dev"
  ]
};

export default nextConfig;
