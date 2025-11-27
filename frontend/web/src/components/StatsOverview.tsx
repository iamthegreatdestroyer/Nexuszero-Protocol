"use client";

import { motion } from "framer-motion";
import { TrendingUp, Shield, Zap, Globe } from "lucide-react";

interface StatCard {
  title: string;
  value: string;
  change: string;
  changeType: "positive" | "negative" | "neutral";
  icon: React.ReactNode;
}

const stats: StatCard[] = [
  {
    title: "Total Proofs Generated",
    value: "1,247,893",
    change: "+12.5%",
    changeType: "positive",
    icon: <Shield className="w-6 h-6" />,
  },
  {
    title: "Privacy Score",
    value: "94.2",
    change: "+2.1%",
    changeType: "positive",
    icon: <TrendingUp className="w-6 h-6" />,
  },
  {
    title: "Avg Generation Time",
    value: "847ms",
    change: "-15.3%",
    changeType: "positive",
    icon: <Zap className="w-6 h-6" />,
  },
  {
    title: "Chains Supported",
    value: "12",
    change: "+2 new",
    changeType: "neutral",
    icon: <Globe className="w-6 h-6" />,
  },
];

export function StatsOverview() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.title}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className="glass rounded-xl p-6 glass-hover cursor-pointer group"
        >
          <div className="flex items-start justify-between mb-4">
            <div className="p-3 rounded-xl bg-nexus-primary/20 text-nexus-primary group-hover:bg-nexus-primary group-hover:text-white transition-colors">
              {stat.icon}
            </div>
            <span
              className={`text-sm font-medium px-2 py-1 rounded-full ${
                stat.changeType === "positive"
                  ? "bg-green-500/20 text-green-400"
                  : stat.changeType === "negative"
                  ? "bg-red-500/20 text-red-400"
                  : "bg-slate-500/20 text-slate-400"
              }`}
            >
              {stat.change}
            </span>
          </div>
          <p className="text-slate-400 text-sm mb-1">{stat.title}</p>
          <p className="text-3xl font-bold">{stat.value}</p>
        </motion.div>
      ))}
    </div>
  );
}
