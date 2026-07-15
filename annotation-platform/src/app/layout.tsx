import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Guardrail Annotation Platform",
  description: "Quantitative dashboards and qualitative annotation for guardrail evaluation results",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-slate-50 text-slate-900">
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto max-w-6xl px-6 py-3 flex items-center gap-6">
            <Link href="/" className="font-semibold tracking-tight">
              Guardrail Annotation Platform
            </Link>
            <nav className="flex gap-4 text-sm text-slate-600">
              <Link href="/humanitarian" className="hover:text-slate-900">Humanitarian</Link>
              <Link href="/financial" className="hover:text-slate-900">Financial</Link>
              <Link href="/cybersecurity" className="hover:text-slate-900">Cybersecurity</Link>
              <Link href="/compare" className="hover:text-slate-900">Compare</Link>
            </nav>
          </div>
        </header>
        <main className="flex-1 mx-auto w-full max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
