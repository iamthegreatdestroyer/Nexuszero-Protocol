"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Eye,
  EyeOff,
  Cpu,
  Clock,
  CheckCircle,
  Fingerprint,
  Lock,
  Unlock,
} from "lucide-react";

interface ProofStage {
  name: string;
  status: "pending" | "active" | "complete";
  duration?: number;
}

interface ProofData {
  id: string;
  privacyLevel: number;
  stages: ProofStage[];
  proofSize: number;
  generationTime: number;
  verified: boolean;
}

export function ProofVisualizer() {
  const [activeProof, setActiveProof] = useState<ProofData | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const simulateProofGeneration = () => {
    setIsGenerating(true);
    setShowDetails(true);

    const stages: ProofStage[] = [
      { name: "Initialize Circuit", status: "pending" },
      { name: "Compute Witness", status: "pending" },
      { name: "Generate Commitments", status: "pending" },
      { name: "Create ZK Proof", status: "pending" },
      { name: "Verify Locally", status: "pending" },
    ];

    setActiveProof({
      id: `proof-${Date.now()}`,
      privacyLevel: 3,
      stages,
      proofSize: 0,
      generationTime: 0,
      verified: false,
    });

    let currentStage = 0;
    const interval = setInterval(() => {
      if (currentStage >= stages.length) {
        clearInterval(interval);
        setIsGenerating(false);
        setActiveProof((prev) =>
          prev
            ? {
                ...prev,
                verified: true,
                proofSize: 1247,
                generationTime: 847,
              }
            : null
        );
        return;
      }

      setActiveProof((prev) => {
        if (!prev) return null;
        const newStages = [...prev.stages];
        if (currentStage > 0) {
          newStages[currentStage - 1] = {
            ...newStages[currentStage - 1],
            status: "complete",
            duration: 150 + Math.random() * 100,
          };
        }
        newStages[currentStage] = {
          ...newStages[currentStage],
          status: "active",
        };
        return { ...prev, stages: newStages };
      });

      currentStage++;
    }, 500);
  };

  return (
    <div className="glass rounded-2xl p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold flex items-center gap-3">
          <Fingerprint className="w-6 h-6 text-nexus-primary" />
          Proof Visualizer
        </h2>
        <button
          onClick={() => setShowDetails(!showDetails)}
          className="flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
        >
          {showDetails ? (
            <EyeOff className="w-4 h-4" />
          ) : (
            <Eye className="w-4 h-4" />
          )}
          <span className="text-sm">
            {showDetails ? "Hide" : "Show"} Details
          </span>
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Proof Generation */}
        <div className="p-6 rounded-xl bg-white/5">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            Proof Generation
          </h3>

          {activeProof && showDetails ? (
            <div className="space-y-4">
              {activeProof.stages.map((stage, index) => (
                <motion.div
                  key={stage.name}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className={`flex items-center justify-between p-3 rounded-lg transition-colors ${
                    stage.status === "active"
                      ? "bg-nexus-primary/20 border border-nexus-primary/50"
                      : stage.status === "complete"
                      ? "bg-green-500/10"
                      : "bg-white/5"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {stage.status === "active" && (
                      <div className="w-4 h-4 rounded-full border-2 border-nexus-primary border-t-transparent animate-spin" />
                    )}
                    {stage.status === "complete" && (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    )}
                    {stage.status === "pending" && (
                      <div className="w-4 h-4 rounded-full border-2 border-slate-600" />
                    )}
                    <span
                      className={
                        stage.status === "active"
                          ? "text-white"
                          : "text-slate-400"
                      }
                    >
                      {stage.name}
                    </span>
                  </div>
                  {stage.duration && (
                    <span className="text-sm text-slate-500">
                      {stage.duration.toFixed(0)}ms
                    </span>
                  )}
                </motion.div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-slate-400 mb-4">
                Generate a proof to see the visualization
              </p>
            </div>
          )}

          <button
            onClick={simulateProofGeneration}
            disabled={isGenerating}
            className="w-full mt-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-500 font-semibold hover:opacity-90 transition-opacity disabled:opacity-50 flex items-center justify-center gap-2"
          >
            {isGenerating ? (
              <>
                <div className="w-5 h-5 rounded-full border-2 border-white border-t-transparent animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Activity className="w-5 h-5" />
                Generate Proof
              </>
            )}
          </button>
        </div>

        {/* Proof Details */}
        <div className="p-6 rounded-xl bg-white/5">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Lock className="w-5 h-5 text-purple-400" />
            Proof Details
          </h3>

          {activeProof?.verified ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="space-y-4"
            >
              <div className="flex items-center justify-between p-4 rounded-lg bg-green-500/10 border border-green-500/30">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-6 h-6 text-green-400" />
                  <div>
                    <p className="font-semibold text-green-400">
                      Proof Verified
                    </p>
                    <p className="text-sm text-slate-400">
                      Zero-knowledge proof is valid
                    </p>
                  </div>
                </div>
                <Unlock className="w-6 h-6 text-green-400" />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-white/5">
                  <p className="text-sm text-slate-400">Proof Size</p>
                  <p className="text-2xl font-bold">
                    {activeProof.proofSize} bytes
                  </p>
                </div>
                <div className="p-4 rounded-lg bg-white/5">
                  <p className="text-sm text-slate-400">Generation Time</p>
                  <p className="text-2xl font-bold">
                    {activeProof.generationTime}ms
                  </p>
                </div>
              </div>

              <div className="p-4 rounded-lg bg-white/5">
                <p className="text-sm text-slate-400 mb-2">Proof ID</p>
                <p className="font-mono text-sm break-all">{activeProof.id}</p>
              </div>

              {/* Visual Proof Representation */}
              <div className="p-4 rounded-lg bg-white/5">
                <p className="text-sm text-slate-400 mb-3">Proof Structure</p>
                <div className="grid grid-cols-8 gap-1">
                  {Array.from({ length: 64 }).map((_, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, scale: 0 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: i * 0.02 }}
                      className="aspect-square rounded-sm"
                      style={{
                        backgroundColor: `hsl(${(i * 5 + 240) % 360}, 70%, ${
                          50 + Math.random() * 20
                        }%)`,
                        opacity: 0.6 + Math.random() * 0.4,
                      }}
                    />
                  ))}
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-slate-400">
              <Lock className="w-12 h-12 mb-4 opacity-50" />
              <p>No proof generated yet</p>
              <p className="text-sm">Click "Generate Proof" to start</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
