"use client";

import { ConnectButton } from "@rainbow-me/rainbowkit";
import { PrivacyDashboard } from "@/components/PrivacyDashboard";
import { TransactionHistory } from "@/components/TransactionHistory";
import { ProofVisualizer } from "@/components/ProofVisualizer";
import { StatsOverview } from "@/components/StatsOverview";
import { NetworkStatus } from "@/components/NetworkStatus";
import { Shield, Zap, Link2, Lock } from "lucide-react";

export default function Home() {
  return (
    <main className="min-h-screen">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-nexus-gradient flex items-center justify-center">
                <Shield className="w-6 h-6 text-white" />
              </div>
              <h1 className="text-xl font-bold text-gradient">NexusZero</h1>
            </div>

            <div className="hidden md:flex items-center gap-8">
              <NavLink href="#dashboard" icon={<Lock className="w-4 h-4" />}>
                Privacy
              </NavLink>
              <NavLink href="#transactions" icon={<Zap className="w-4 h-4" />}>
                Transactions
              </NavLink>
              <NavLink href="#proofs" icon={<Link2 className="w-4 h-4" />}>
                Proofs
              </NavLink>
            </div>

            <ConnectButton
              showBalance={false}
              chainStatus="icon"
              accountStatus="address"
            />
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-16 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <h2 className="text-5xl md:text-6xl font-bold mb-6">
            <span className="text-gradient">Privacy-First</span>
            <br />
            Cross-Chain Transactions
          </h2>
          <p className="text-xl text-slate-400 max-w-2xl mx-auto mb-8">
            Execute private transactions across any blockchain with
            zero-knowledge proofs. Your data, your control, your sovereignty.
          </p>
          <div className="flex flex-wrap gap-4 justify-center">
            <button className="px-8 py-3 rounded-xl bg-nexus-gradient font-semibold hover:opacity-90 transition-opacity">
              Start Transacting
            </button>
            <button className="px-8 py-3 rounded-xl glass glass-hover font-semibold">
              Learn More
            </button>
          </div>
        </div>
      </section>

      {/* Stats Overview */}
      <section className="py-8 px-4">
        <div className="max-w-7xl mx-auto">
          <StatsOverview />
        </div>
      </section>

      {/* Main Dashboard Grid */}
      <section id="dashboard" className="py-8 px-4">
        <div className="max-w-7xl mx-auto">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PrivacyDashboard />
            </div>
            <div>
              <NetworkStatus />
            </div>
          </div>
        </div>
      </section>

      {/* Transaction History */}
      <section id="transactions" className="py-8 px-4">
        <div className="max-w-7xl mx-auto">
          <TransactionHistory />
        </div>
      </section>

      {/* Proof Visualizer */}
      <section id="proofs" className="py-8 px-4 pb-16">
        <div className="max-w-7xl mx-auto">
          <ProofVisualizer />
        </div>
      </section>

      {/* Footer */}
      <footer className="glass border-t border-white/10 py-8 px-4">
        <div className="max-w-7xl mx-auto text-center text-slate-400">
          <p>© 2025 NexusZero Protocol. All rights reserved.</p>
          <p className="text-sm mt-2">Privacy is a right, not a privilege.</p>
        </div>
      </footer>
    </main>
  );
}

function NavLink({
  href,
  icon,
  children,
}: {
  href: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <a
      href={href}
      className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
    >
      {icon}
      {children}
    </a>
  );
}
