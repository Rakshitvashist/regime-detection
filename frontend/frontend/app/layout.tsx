import "./globals.css";
import { JetBrains_Mono } from "next/font/google";
import { Retune } from "retune";

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  display: "swap",
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={jetbrains.variable}>
      <body className="font-mono antialiased bg-[#fafafa] text-zinc-900">
        <Retune />
        {children}
      </body>
    </html>
  );
}