import { NEOTracker } from "@/components/NEOTracker";
import { SystemStatus } from "@/components/SystemStatus";
import { useEffect, useState } from "react";
import { getNEOData, NEOData } from "./lib/neoService";

const Index = () => {
  const [neoData, setNeoData] = useState<NEOData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await getNEOData();
        setNeoData(data);
      } catch (error) {
        console.error("Error fetching NEO data:", error);
        setError("Failed to fetch NEO data. Please check if the backend is running.");
      } finally {
        setLoading(false);
      }
    };
    fetchData();
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
        </div>
      </header>

      {/* Scanning Line Effect */}
      <div className="relative">
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="w-full h-0.5 bg-primary/30 scan-animation" />
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-4">
          <NEOTracker neoData={neoData} loading={loading} />
        </div>

        {/* Right Column */}
        <div className="space-y-4">
          <SystemStatus neoData={neoData} loading={loading}/>
        </div>
      </div>
    </div>
  );
};

export default Index;
