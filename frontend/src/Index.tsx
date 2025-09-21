import { HowItWorks } from "@/components/HowItWorks";
import { NEOTracker } from "@/components/NEOTracker";
import SpaceVisualization from "@/components/SpaceVisualization";
import { SystemStatus } from "@/components/SystemStatus";
import { useEffect, useState } from "react";
import { getMockNEOData, NEOData } from "./lib/neoService";

const Index = () => {
  const [neoData, setNeoData] = useState<NEOData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadMockData = () => {
      try {
        setLoading(true);
        setError(null);
        // Using mock data for PHA refactor
        const data = getMockNEOData();
        setNeoData(data);
      } catch (error) {
        console.error("Error loading mock NEO data:", error);
        setError("Failed to load NEO data.");
      } finally {
        setLoading(false);
      }
    };

    // Simulate async loading for loading state demonstration
    setTimeout(loadMockData, 1000);
  }, []);

  return (
    <div className="min-h-screen bg-background terminal-grid">
      {/* Header */}
      <header className="border-b border-primary/30 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold glow-primary">STAR SHIELD</h1>
            <p className="text-sm text-muted-foreground">
              Near Earth Object Tracking & Threat Assessment System
            </p>
          </div>
          <HowItWorks neoData={neoData} loading={loading} />
        </div>
      </header>

      {/* Scanning Line Effect */}
      <div className="relative">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="w-full h-0.5 bg-primary/30 scan-animation" />
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left Column */}
        <div className="lg:col-span-1 space-y-4">
          <SpaceVisualization neoData={neoData} className="h-full" />
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          <SystemStatus neoData={neoData} loading={loading} />
          <NEOTracker neoData={neoData} loading={loading} />
        </div>
      </div>
    </div>
  );
};

export default Index;
