import { HowItWorks } from "@/components/HowItWorks";
import { NEOTracker } from "@/components/NEOTracker";
import { RiskAssessmentMatrix } from "@/components/RiskAssessmentMatrix";
import SpaceVisualization from "@/components/SpaceVisualization";
import { SystemStatus } from "@/components/SystemStatus";
import { useEffect, useState } from "react";
import { getMockNEOData, getNEOData, NEOData } from "./lib/neoService";

const Index = () => {
  const [neoData, setNeoData] = useState<NEOData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedNEO, setSelectedNEO] = useState<string | null>(null);
  const [sizeFilter, setSizeFilter] = useState<string | null>(null);
  const [velocityFilter, setVelocityFilter] = useState<string | null>(null);

  // Handle matrix cell click for filtering
  const handleMatrixFilter = (size: string, velocity: string) => {
    setSizeFilter(size);
    setVelocityFilter(velocity);
  };

  // Clear filters
  const clearFilters = () => {
    setSizeFilter(null);
    setVelocityFilter(null);
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await getNEOData();
        setNeoData(data);
      } catch (error) {
        console.error("Error loading mock NEO data:", error);
        setError("Failed to load NEO data.");
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
            <h1 className="text-2xl font-bold glow-primary mb-1">
              STAR SHIELD
            </h1>
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
      <div className="p-4 grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Left Column */}
        <div className="lg:col-span-2 space-y-4">
          <SpaceVisualization
            neoData={neoData}
            setSelectedNEO={setSelectedNEO}
            selectedNEO={selectedNEO}
          />
          <RiskAssessmentMatrix 
            neoData={neoData} 
            loading={loading}
            onFilterSelect={handleMatrixFilter}
            activeFilters={{ size: sizeFilter, velocity: velocityFilter }}
          />
        </div>

        {/* Right Column */}
        <div className="space-y-4 lg:col-span-2">
          <SystemStatus neoData={neoData} loading={loading} />
          <NEOTracker
            neoData={neoData}
            loading={loading}
            selectedNEO={selectedNEO}
            setSelectedNEO={setSelectedNEO}
            sizeFilter={sizeFilter}
            velocityFilter={velocityFilter}
            onClearFilters={clearFilters}
          />
        </div>
      </div>
    </div>
  );
};

export default Index;
