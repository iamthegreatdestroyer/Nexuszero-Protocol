"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Shield, ChevronDown, Info, Zap } from "lucide-react";

export interface PrivacyLevel {
  level: number;
  name: string;
  description: string;
  securityBits: number;
  color: string;
  features: string[];
}

const PRIVACY_LEVELS: PrivacyLevel[] = [
  {
    level: 0,
    name: "Transparent",
    description: "Full visibility - suitable for public transactions",
    securityBits: 0,
    color: "#94a3b8",
    features: [
      "Public transaction",
      "Full auditability",
      "No privacy overhead",
    ],
  },
  {
    level: 1,
    name: "Pseudonymous",
    description: "Address-based privacy without identity linkage",
    securityBits: 64,
    color: "#60a5fa",
    features: ["Unlinkable addresses", "Basic privacy", "Low gas costs"],
  },
  {
    level: 2,
    name: "Confidential",
    description: "Hidden amounts with visible participants",
    securityBits: 128,
    color: "#34d399",
    features: ["Hidden amounts", "Pedersen commitments", "Range proofs"],
  },
  {
    level: 3,
    name: "Private",
    description: "Hidden amounts and participants with compliance support",
    securityBits: 192,
    color: "#a78bfa",
    features: [
      "Hidden amounts & parties",
      "Compliance proofs",
      "Selective disclosure",
    ],
  },
  {
    level: 4,
    name: "Anonymous",
    description: "Full transaction privacy with mixing",
    securityBits: 256,
    color: "#f472b6",
    features: ["Ring signatures", "Stealth addresses", "Transaction mixing"],
  },
  {
    level: 5,
    name: "Sovereign",
    description: "Maximum privacy with post-quantum protection",
    securityBits: 384,
    color: "#fbbf24",
    features: ["Post-quantum secure", "Zero metadata", "Maximum entropy"],
  },
];

export function PrivacyDashboard() {
  const [selectedLevel, setSelectedLevel] = useState<number>(3);
  const [isExpanded, setIsExpanded] = useState(false);

  const currentLevel = PRIVACY_LEVELS[selectedLevel];

  return (
    <div className="glass rounded-2xl p-6 glow-box">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold flex items-center gap-3">
          <Shield className="w-6 h-6 text-nexus-primary" />
          Privacy Control
        </h2>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
        >
          <Info className="w-4 h-4" />
          <span className="text-sm">Details</span>
          <ChevronDown
            className={`w-4 h-4 transition-transform ${
              isExpanded ? "rotate-180" : ""
            }`}
          />
        </button>
      </div>

      {/* Privacy Level Selector */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <span className="text-sm text-slate-400">Select Privacy Level</span>
          <span
            className="text-sm font-semibold px-3 py-1 rounded-full"
            style={{
              backgroundColor: `${currentLevel.color}20`,
              color: currentLevel.color,
            }}
          >
            {currentLevel.securityBits}-bit security
          </span>
        </div>

        {/* Slider Track */}
        <div className="relative h-3 bg-slate-800 rounded-full mb-4">
          <motion.div
            className="absolute h-full rounded-full"
            style={{
              background: `linear-gradient(90deg, ${PRIVACY_LEVELS[0].color}, ${currentLevel.color})`,
              width: `${(selectedLevel / 5) * 100}%`,
            }}
            animate={{ width: `${(selectedLevel / 5) * 100}%` }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          />

          {/* Level Markers */}
          <div className="absolute inset-0 flex justify-between px-1">
            {PRIVACY_LEVELS.map((level, index) => (
              <button
                key={level.level}
                onClick={() => setSelectedLevel(index)}
                className="relative group"
              >
                <div
                  className={`w-3 h-3 rounded-full border-2 transition-all ${
                    index <= selectedLevel
                      ? "border-transparent"
                      : "border-slate-600 bg-slate-800"
                  }`}
                  style={{
                    backgroundColor:
                      index <= selectedLevel ? level.color : undefined,
                  }}
                />
                <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                  {level.name}
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Current Level Display */}
      <motion.div
        key={selectedLevel}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="p-6 rounded-xl"
        style={{
          backgroundColor: `${currentLevel.color}10`,
          borderColor: `${currentLevel.color}30`,
        }}
      >
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3
              className="text-xl font-bold"
              style={{ color: currentLevel.color }}
            >
              Level {currentLevel.level}: {currentLevel.name}
            </h3>
            <p className="text-slate-400 mt-1">{currentLevel.description}</p>
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <Zap className="w-4 h-4" />
            <span className="text-sm">~{0.001 * (selectedLevel + 1)} ETH</span>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {currentLevel.features.map((feature) => (
            <span
              key={feature}
              className="px-3 py-1 text-sm rounded-full bg-white/5 border border-white/10"
            >
              {feature}
            </span>
          ))}
        </div>
      </motion.div>

      {/* Expanded Details */}
      {isExpanded && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="mt-6 pt-6 border-t border-white/10"
        >
          <h4 className="text-lg font-semibold mb-4">All Privacy Levels</h4>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {PRIVACY_LEVELS.map((level) => (
              <button
                key={level.level}
                onClick={() => setSelectedLevel(level.level)}
                className={`p-4 rounded-xl text-left transition-all ${
                  selectedLevel === level.level
                    ? "ring-2"
                    : "bg-white/5 hover:bg-white/10"
                }`}
                style={{
                  // @ts-ignore - CSS variable for ring color
                  "--tw-ring-color":
                    selectedLevel === level.level ? level.color : undefined,
                  backgroundColor:
                    selectedLevel === level.level
                      ? `${level.color}10`
                      : undefined,
                }}
              >
                <div className="flex items-center gap-2 mb-2">
                  <div
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: level.color }}
                  />
                  <span className="font-semibold">{level.name}</span>
                </div>
                <p className="text-xs text-slate-400">
                  {level.securityBits}-bit security
                </p>
              </button>
            ))}
          </div>
        </motion.div>
      )}

      {/* Action Button */}
      <button className="w-full mt-6 py-4 rounded-xl bg-nexus-gradient font-semibold hover:opacity-90 transition-opacity flex items-center justify-center gap-2">
        <Shield className="w-5 h-5" />
        Apply Privacy Level
      </button>
    </div>
  );
}
