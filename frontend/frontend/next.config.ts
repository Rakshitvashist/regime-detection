import type { NextConfig } from "next";

let githubRepo = "";
if (process.env.GITHUB_REPOSITORY) {
  githubRepo = "/" + process.env.GITHUB_REPOSITORY.split("/")[1];
}

const nextConfig: any = {
  output: "export",
  basePath: githubRepo,
  assetPrefix: githubRepo,
  eslint: { ignoreDuringBuilds: true },
  typescript: { ignoreBuildErrors: true },
  images: { unoptimized: true },
};

export default nextConfig;
