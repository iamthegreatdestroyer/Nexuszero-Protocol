"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Server,
  Wifi,
  WifiOff,
  CheckCircle,
  AlertCircle,
  Clock,
} from "lucide-react";

interface NetworkNode {
  id: string;
  name: string;
  status: "online" | "offline" | "syncing";
  latency: number;
  proofCount: number;
}

const MOCK_NODES: NetworkNode[] = [
  {
    id: "1",
    name: "Ethereum Mainnet",
    status: "online",
    latency: 45,
    proofCount: 12543,
  },
  { id: "2", name: "Polygon", status: "online", latency: 23, proofCount: 8921 },
  {
    id: "3",
    name: "Arbitrum",
    status: "syncing",
    latency: 78,
    proofCount: 5432,
  },
  {
    id: "4",
    name: "Optimism",
    status: "online",
    latency: 52,
    proofCount: 3211,
  },
  { id: "5", name: "Base", status: "offline", latency: 0, proofCount: 1234 },
];

export function NetworkStatus() {
  const [nodes, setNodes] = useState<NetworkNode[]>(MOCK_NODES);
  const [overallStatus, setOverallStatus] = useState<
    "healthy" | "degraded" | "offline"
  >("healthy");

  useEffect(() => {
    const onlineCount = nodes.filter((n) => n.status === "online").length;
    const totalCount = nodes.length;

    if (onlineCount === totalCount) {
      setOverallStatus("healthy");
    } else if (onlineCount > 0) {
      setOverallStatus("degraded");
    } else {
      setOverallStatus("offline");
    }
  }, [nodes]);

  const getStatusIcon = (status: NetworkNode["status"]) => {
    switch (status) {
      case "online":
        return <Wifi className="w-4 h-4 text-green-400" />;
      case "syncing":
        return <Clock className="w-4 h-4 text-yellow-400 animate-pulse" />;
      case "offline":
        return <WifiOff className="w-4 h-4 text-red-400" />;
    }
  };

  const getStatusColor = (status: NetworkNode["status"]) => {
    switch (status) {
      case "online":
        return "bg-green-400";
      case "syncing":
        return "bg-yellow-400";
      case "offline":
        return "bg-red-400";
    }
  };

  return (
    <div className="glass rounded-2xl p-6 h-full">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold flex items-center gap-3">
          <Server className="w-5 h-5 text-nexus-primary" />
          Network Status
        </h2>
        <div
          className={`flex items-center gap-2 px-3 py-1 rounded-full text-sm ${
            overallStatus === "healthy"
              ? "bg-green-500/20 text-green-400"
              : overallStatus === "degraded"
              ? "bg-yellow-500/20 text-yellow-400"
              : "bg-red-500/20 text-red-400"
          }`}
        >
          {overallStatus === "healthy" && <CheckCircle className="w-4 h-4" />}
          {overallStatus === "degraded" && <AlertCircle className="w-4 h-4" />}
          {overallStatus === "offline" && <WifiOff className="w-4 h-4" />}
          <span className="capitalize">{overallStatus}</span>
        </div>
      </div>

      {/* Network Activity Indicator */}
      <div className="mb-6 p-4 rounded-xl bg-white/5">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-slate-400">Network Activity</span>
          <Activity className="w-4 h-4 text-nexus-primary" />
        </div>
        <div className="h-8 flex items-end gap-1">
          {Array.from({ length: 20 }).map((_, i) => (
            <motion.div
              key={i}
              className="flex-1 bg-nexus-primary/50 rounded-t"
              initial={{ height: 0 }}
              animate={{
                height: `${20 + Math.random() * 80}%`,
              }}
              transition={{
                duration: 0.5,
                delay: i * 0.05,
                repeat: Infinity,
                repeatType: "reverse",
                repeatDelay: Math.random() * 2,
              }}
            />
          ))}
        </div>
      </div>

      {/* Node List */}
      <div className="space-y-3">
        {nodes.map((node, index) => (
          <motion.div
            key={node.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            className="flex items-center justify-between p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-colors"
          >
            <div className="flex items-center gap-3">
              <div
                className={`w-2 h-2 rounded-full ${getStatusColor(
                  node.status
                )}`}
              />
              <div>
                <p className="font-medium text-sm">{node.name}</p>
                <p className="text-xs text-slate-500">
                  {node.proofCount.toLocaleString()} proofs
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {node.status !== "offline" && (
                <span className="text-xs text-slate-400">{node.latency}ms</span>
              )}
              {getStatusIcon(node.status)}
            </div>
          </motion.div>
        ))}
      </div>

      {/* Prover Network Stats */}
      <div className="mt-6 pt-6 border-t border-white/10">
        <h3 className="text-sm font-semibold text-slate-400 mb-4">
          Prover Network
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-3 rounded-lg bg-white/5 text-center">
            <p className="text-2xl font-bold text-nexus-primary">247</p>
            <p className="text-xs text-slate-400">Active Provers</p>
          </div>
          <div className="p-3 rounded-lg bg-white/5 text-center">
            <p className="text-2xl font-bold text-cyan-400">1.2M</p>
            <p className="text-xs text-slate-400">Total Capacity</p>
          </div>
        </div>
      </div>
    </div>
  );
}
