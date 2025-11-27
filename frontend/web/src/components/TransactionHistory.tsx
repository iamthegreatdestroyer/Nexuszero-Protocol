"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ArrowUpRight,
  ArrowDownLeft,
  Clock,
  CheckCircle2,
  XCircle,
  Shield,
  ExternalLink,
  Filter,
  Search,
} from "lucide-react";

interface Transaction {
  id: string;
  type: "send" | "receive" | "bridge";
  amount: string;
  token: string;
  privacyLevel: number;
  status: "pending" | "confirmed" | "failed";
  timestamp: Date;
  txHash: string;
  chain: string;
  counterparty?: string;
}

const MOCK_TRANSACTIONS: Transaction[] = [
  {
    id: "1",
    type: "send",
    amount: "1.5",
    token: "ETH",
    privacyLevel: 3,
    status: "confirmed",
    timestamp: new Date(Date.now() - 1000 * 60 * 15),
    txHash: "0x1234...5678",
    chain: "Ethereum",
    counterparty: "0xabcd...efgh",
  },
  {
    id: "2",
    type: "receive",
    amount: "500",
    token: "USDC",
    privacyLevel: 4,
    status: "confirmed",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 2),
    txHash: "0x5678...9abc",
    chain: "Polygon",
    counterparty: "0x9876...5432",
  },
  {
    id: "3",
    type: "bridge",
    amount: "0.5",
    token: "ETH",
    privacyLevel: 5,
    status: "pending",
    timestamp: new Date(Date.now() - 1000 * 60 * 5),
    txHash: "0xdef0...1234",
    chain: "Ethereum → Arbitrum",
  },
  {
    id: "4",
    type: "send",
    amount: "100",
    token: "USDT",
    privacyLevel: 2,
    status: "failed",
    timestamp: new Date(Date.now() - 1000 * 60 * 60 * 24),
    txHash: "0xfade...cafe",
    chain: "Ethereum",
    counterparty: "0xdead...beef",
  },
];

const PRIVACY_COLORS = [
  "#94a3b8",
  "#60a5fa",
  "#34d399",
  "#a78bfa",
  "#f472b6",
  "#fbbf24",
];
const PRIVACY_NAMES = [
  "Transparent",
  "Pseudonymous",
  "Confidential",
  "Private",
  "Anonymous",
  "Sovereign",
];

export function TransactionHistory() {
  const [transactions] = useState<Transaction[]>(MOCK_TRANSACTIONS);
  const [filter, setFilter] = useState<"all" | "send" | "receive" | "bridge">(
    "all"
  );
  const [searchQuery, setSearchQuery] = useState("");

  const filteredTransactions = transactions.filter((tx) => {
    if (filter !== "all" && tx.type !== filter) return false;
    if (
      searchQuery &&
      !tx.txHash.toLowerCase().includes(searchQuery.toLowerCase())
    )
      return false;
    return true;
  });

  const getStatusIcon = (status: Transaction["status"]) => {
    switch (status) {
      case "confirmed":
        return <CheckCircle2 className="w-4 h-4 text-green-400" />;
      case "pending":
        return <Clock className="w-4 h-4 text-yellow-400 animate-pulse" />;
      case "failed":
        return <XCircle className="w-4 h-4 text-red-400" />;
    }
  };

  const getTypeIcon = (type: Transaction["type"]) => {
    switch (type) {
      case "send":
        return <ArrowUpRight className="w-5 h-5 text-red-400" />;
      case "receive":
        return <ArrowDownLeft className="w-5 h-5 text-green-400" />;
      case "bridge":
        return <ExternalLink className="w-5 h-5 text-blue-400" />;
    }
  };

  const formatTimeAgo = (date: Date) => {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
  };

  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <h2 className="text-2xl font-bold">Transaction History</h2>

        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
              type="text"
              placeholder="Search by hash..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-2 focus:ring-nexus-primary text-sm"
            />
          </div>

          {/* Filter */}
          <div className="flex items-center gap-2 bg-white/5 rounded-lg p-1">
            {(["all", "send", "receive", "bridge"] as const).map((type) => (
              <button
                key={type}
                onClick={() => setFilter(type)}
                className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                  filter === type
                    ? "bg-nexus-primary text-white"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {type.charAt(0).toUpperCase() + type.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Transaction List */}
      <div className="space-y-3">
        <AnimatePresence mode="popLayout">
          {filteredTransactions.map((tx) => (
            <motion.div
              key={tx.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
              layout
              className="flex items-center justify-between p-4 rounded-xl bg-white/5 hover:bg-white/10 transition-colors cursor-pointer group"
            >
              <div className="flex items-center gap-4">
                {/* Type Icon */}
                <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center">
                  {getTypeIcon(tx.type)}
                </div>

                {/* Transaction Details */}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-semibold">
                      {tx.type === "send"
                        ? "Sent"
                        : tx.type === "receive"
                        ? "Received"
                        : "Bridged"}
                    </span>
                    <span className="text-slate-400">
                      {tx.amount} {tx.token}
                    </span>
                    {tx.counterparty && (
                      <>
                        <span className="text-slate-500">→</span>
                        <span className="text-sm text-slate-400 font-mono">
                          {tx.counterparty}
                        </span>
                      </>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-1 text-sm text-slate-500">
                    <span>{tx.chain}</span>
                    <span>•</span>
                    <span className="font-mono">{tx.txHash}</span>
                    <span>•</span>
                    <span>{formatTimeAgo(tx.timestamp)}</span>
                  </div>
                </div>
              </div>

              <div className="flex items-center gap-4">
                {/* Privacy Level */}
                <div
                  className="flex items-center gap-2 px-3 py-1 rounded-full text-sm"
                  style={{
                    backgroundColor: `${PRIVACY_COLORS[tx.privacyLevel]}20`,
                    color: PRIVACY_COLORS[tx.privacyLevel],
                  }}
                >
                  <Shield className="w-3 h-3" />
                  <span>{PRIVACY_NAMES[tx.privacyLevel]}</span>
                </div>

                {/* Status */}
                <div className="flex items-center gap-2">
                  {getStatusIcon(tx.status)}
                  <span
                    className={`text-sm capitalize ${
                      tx.status === "confirmed"
                        ? "text-green-400"
                        : tx.status === "pending"
                        ? "text-yellow-400"
                        : "text-red-400"
                    }`}
                  >
                    {tx.status}
                  </span>
                </div>

                {/* External Link */}
                <ExternalLink className="w-4 h-4 text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity" />
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {filteredTransactions.length === 0 && (
          <div className="text-center py-12 text-slate-400">
            No transactions found
          </div>
        )}
      </div>
    </div>
  );
}
