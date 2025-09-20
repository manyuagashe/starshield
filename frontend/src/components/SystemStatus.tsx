import { NEOData } from "@/lib/neoService";
import { useEffect, useState } from "react";

export const SystemStatus = ({ neoData }: { neoData: NEOData[] }) => {
  const [systemTime, setSystemTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setSystemTime(new Date());
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  const immediateThreats = neoData.filter(
    (neo) => neo.threatLevel === "CRITICAL"
  ).length;
  const highPriorityThreats = neoData.filter((neo) =>
    ["HIGH", "MEDIUM"].includes(neo.threatLevel)
  ).length;

  return (
    <div className="command-panel">
      <h2 className="text-lg font-bold glow-primary mb-4">SYSTEM STATUS</h2>

      <div className="space-y-4">
        <div>
          <div className="text-xs text-muted-foreground">SYSTEM TIME (UTC)</div>
          <div className="text-xl font-mono text-primary glow-primary">
            {systemTime.toISOString().replace("T", " ").slice(0, 19)}
          </div>
        </div>

        <div>
          <div className="text-xs text-muted-foreground mb-2">
            THREAT ASSESSMENT
          </div>
          <div className="text-sm">
            <div className="flex justify-between">
              <span>MONITORED OBJECTS:</span>
              <span className="text-primary font-mono">{neoData.length}</span>
            </div>
            <div className="flex justify-between">
              <span>HIGH PRIORITY:</span>
              <span className="text-warning font-mono">{highPriorityThreats}</span>
            </div>
            <div className="flex justify-between">
              <span>IMMEDIATE THREATS:</span>
              <span className="text-destructive font-mono">
                {immediateThreats}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
