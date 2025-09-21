import { NEOData } from "@/lib/neoService";
import { useMemo } from "react";
import { ThreatIndicator } from "./ThreatIndicator";

export const NEOTracker = ({
  neoData,
  loading,
}: {
  neoData: NEOData[];
  loading: boolean;
}) => {
  const orderedNEOData = useMemo(() => {
    const threatLevelOrder = { CRITICAL: 4, HIGH: 3, MEDIUM: 2, LOW: 1 };
    return [...neoData].sort((a, b) => {
      const threatComparison =
        threatLevelOrder[b.threatLevel] - threatLevelOrder[a.threatLevel];
      if (threatComparison !== 0) return threatComparison;

      return b.impactProbability - a.impactProbability;
    });
  }, [neoData]);

  const formatDistance = (distance: number) => {
    return (distance * 149.6).toFixed(3); // Convert AU to million km
  };

  const formatTime = (hours: number) => {
    if (hours < 24) return `${hours.toFixed(1)}h`;
    const days = Math.floor(hours / 24);
    const remainingHours = Math.floor(hours % 24);
    return `${days}d ${remainingHours}h`;
  };

  return (
    <div className="command-panel">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold glow-primary">
          NEAR EARTH OBJECT TRACKER
        </h2>
      </div>

      <div className="space-y-2">
        {loading ? (
          <div className="neo-tracking-item">
            <div className="flex items-center justify-center p-8">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                <span className="text-primary glow-primary">
                  Scanning for Near Earth Objects...
                </span>
              </div>
            </div>
          </div>
        ) : orderedNEOData.length === 0 ? (
          <div className="neo-tracking-item">
            <div className="flex items-center justify-center p-8">
              <span className="text-muted-foreground">
                No NEO data available
              </span>
            </div>
          </div>
        ) : (
          orderedNEOData.map((neo) => (
            <div key={neo.id} className="neo-tracking-item">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-3">
                  <ThreatIndicator level={neo.threatLevel} />
                  <span className="font-bold text-primary">{neo.name}</span>
                </div>
                <div className="text-xs text-warning">
                  IMPACT PROB: {neo.impactProbability.toFixed(6)}%
                </div>
              </div>

              <div className="grid grid-cols-4 gap-4 text-xs">
                <div>
                  <span className="text-muted-foreground">SIZE:</span>
                  <div className="text-foreground font-mono">
                    {neo.size.toFixed(1)} m
                  </div>
                </div>
                <div>
                  <span className="text-muted-foreground">DISTANCE:</span>
                  <div className="text-foreground font-mono">
                    {formatDistance(neo.distance)} Mkm
                  </div>
                </div>
                <div>
                  <span className="text-muted-foreground">VELOCITY:</span>
                  <div className="text-foreground font-mono">
                    {neo.velocity.toFixed(1)} km/s
                  </div>
                </div>
                <div>
                  <span className="text-muted-foreground">ETA CLOSEST:</span>
                  <div className="text-foreground font-mono">
                    {formatTime(neo.timeToClosestApproach)}
                  </div>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
