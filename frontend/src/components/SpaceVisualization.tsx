import React, { useMemo, useState } from "react";
import { NEOData } from "../lib/neoService";

interface SpaceVisualizationProps {
  neoData: NEOData[];
  className?: string;
}

interface AsteroidPosition {
  id: string;
  x: number;
  y: number;
  size: number;
  isPHA: boolean;
  data: NEOData;
}

const SpaceVisualization: React.FC<SpaceVisualizationProps> = ({
  neoData,
  className = "",
}) => {
  const [selectedAsteroid, setSelectedAsteroid] = useState<string | null>(null);

  // Visualization dimensions
  const width = 500;
  const height = 500;
  const centerX = width / 2;
  const centerY = height / 2;
  const earthRadius = 20;

  // Calculate normalized positions and sizes for asteroids
  const asteroidPositions = useMemo((): AsteroidPosition[] => {
    if (!neoData.length) return [];

    // Find min/max distances for normalization
    const distances = neoData.map((neo) => neo.distance);
    const minDistance = Math.min(...distances);
    const maxDistance = Math.max(...distances);

    // Find min/max sizes for normalization
    const sizes = neoData.map((neo) => neo.size);
    const minSize = Math.min(...sizes);
    const maxSize = Math.max(...sizes);

    // Available radius range (from Earth edge to visualization edge)
    const minRadius = earthRadius + 15; // Buffer from Earth
    const maxRadius = Math.min(width, height) / 2 - 30; // Buffer from edge

    return neoData.map((neo, index) => {
      // Normalize distance to radius (logarithmic scale for better distribution)
      const normalizedDistance =
        (neo.distance - minDistance) / (maxDistance - minDistance);
      const logDistance = Math.log10(1 + normalizedDistance * 9); // Log scale 1-10
      const radius = minRadius + (logDistance / 1) * (maxRadius - minRadius);

      // Distribute asteroids in a circle around Earth
      // Use index-based angle with some randomization for natural look
      const baseAngle = (index / neoData.length) * 2 * Math.PI;
      const angleVariation = Math.sin(index * 0.7) * 0.3; // Small variation
      const angle = baseAngle + angleVariation;

      // Calculate position
      const x = centerX + radius * Math.cos(angle);
      const y = centerY + radius * Math.sin(angle);

      // Normalize size (3-15 pixels range)
      const normalizedSize =
        minSize === maxSize
          ? 6
          : 3 + ((neo.size - minSize) / (maxSize - minSize)) * 12;

      return {
        id: neo.id,
        x,
        y,
        size: normalizedSize,
        isPHA: neo.isPHA,
        data: neo,
      };
    });
  }, [neoData, width, height, centerX, centerY, earthRadius]);

  const handleAsteroidClick = (asteroidId: string) => {
    setSelectedAsteroid(selectedAsteroid === asteroidId ? null : asteroidId);
  };

  return (
    <div className={`relative ${className}`}>
      <div className="bg-black rounded-lg border border-green-500/50 overflow-hidden relative">
        {/* Header */}
        <div className="p-3 border-b border-green-500/50 bg-black/80">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-mono font-bold text-green-400 tracking-wider">
                RADAR SCOPE
              </h3>
              <p className="text-xs text-green-300 font-mono">
                TACTICAL DISPLAY - NEO TRACKING
              </p>
            </div>
            <div className="text-xs font-mono text-green-400">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span>ACTIVE</span>
              </div>
            </div>
          </div>
        </div>

        {/* Radar Display */}
        <div className="p-4 relative">
          {/* Scanlines overlay */}
          <div
            className="absolute inset-0 pointer-events-none z-10"
            style={{
              backgroundImage: `repeating-linear-gradient(
                0deg,
                transparent,
                transparent 2px,
                rgba(34, 197, 94, 0.03) 2px,
                rgba(34, 197, 94, 0.03) 4px
              )`,
            }}
          />

          <svg
            width={width}
            height={height}
            className="rounded-lg bg-black relative"
            style={{
              filter: "drop-shadow(0 0 10px rgba(34, 197, 94, 0.3))",
              background:
                "radial-gradient(circle at center, #001a00 0%, #000000 70%)",
            }}
          >
            {/* Radar sweep animation */}
            <defs>
              <linearGradient id="radarSweep" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="rgba(34, 197, 94, 0)" />
                <stop offset="70%" stopColor="rgba(34, 197, 94, 0.1)" />
                <stop offset="90%" stopColor="rgba(34, 197, 94, 0.3)" />
                <stop offset="100%" stopColor="rgba(34, 197, 94, 0.5)" />
              </linearGradient>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Radar sweep line */}
            <line
              x1={centerX}
              y1={centerY}
              x2={centerX + Math.min(width, height) / 2 - 20}
              y2={centerY}
              stroke="url(#radarSweep)"
              strokeWidth="3"
              filter="url(#glow)"
              className="animate-spin"
              style={{
                transformOrigin: `${centerX}px ${centerY}px`,
                animationDuration: "4s",
              }}
            />

            {/* Radar rings (range circles) */}
            {[0.25, 0.5, 0.75, 1.0].map((factor, i) => (
              <circle
                key={`radar-ring-${i}`}
                cx={centerX}
                cy={centerY}
                r={
                  earthRadius +
                  15 +
                  factor * (Math.min(width, height) / 2 - earthRadius - 45)
                }
                fill="none"
                stroke="rgba(34, 197, 94, 0.3)"
                strokeWidth="1"
                strokeDasharray="3,2"
                filter="url(#glow)"
              />
            ))}

            {/* Range markers */}
            {[0, 90, 180, 270].map((angle, i) => (
              <g key={`range-marker-${i}`}>
                <line
                  x1={centerX + earthRadius * Math.cos((angle * Math.PI) / 180)}
                  y1={centerY + earthRadius * Math.sin((angle * Math.PI) / 180)}
                  x2={
                    centerX +
                    (Math.min(width, height) / 2 - 20) *
                      Math.cos((angle * Math.PI) / 180)
                  }
                  y2={
                    centerY +
                    (Math.min(width, height) / 2 - 20) *
                      Math.sin((angle * Math.PI) / 180)
                  }
                  stroke="rgba(34, 197, 94, 0.2)"
                  strokeWidth="1"
                  strokeDasharray="1,3"
                />
                <text
                  x={
                    centerX +
                    (Math.min(width, height) / 2 - 10) *
                      Math.cos((angle * Math.PI) / 180)
                  }
                  y={
                    centerY +
                    (Math.min(width, height) / 2 - 10) *
                      Math.sin((angle * Math.PI) / 180)
                  }
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="text-xs fill-green-400 font-mono"
                  style={{ fontSize: "10px" }}
                >
                  {angle}°
                </text>
              </g>
            ))}

            {/* Earth (center target) */}
            <circle
              cx={centerX}
              cy={centerY}
              r={earthRadius}
              fill="rgba(34, 197, 94, 0.1)"
              stroke="rgba(34, 197, 94, 0.8)"
              strokeWidth="2"
              filter="url(#glow)"
            />
            <circle
              cx={centerX}
              cy={centerY}
              r={earthRadius - 8}
              fill="none"
              stroke="rgba(34, 197, 94, 0.4)"
              strokeWidth="1"
            />

            {/* Asteroids (radar blips) */}
            {asteroidPositions.map((asteroid) => (
              <g key={asteroid.id}>
                {/* Asteroid radar blip */}
                <circle
                  cx={asteroid.x}
                  cy={asteroid.y}
                  r={asteroid.size}
                  fill={
                    asteroid.isPHA
                      ? "rgba(239, 68, 68, 0.8)"
                      : "rgba(34, 197, 94, 0.6)"
                  }
                  stroke={asteroid.isPHA ? "#ef4444" : "#22c55e"}
                  strokeWidth="1"
                  className={`cursor-pointer transition-all duration-200 ${
                    selectedAsteroid === asteroid.id
                      ? "opacity-100"
                      : "opacity-80"
                  } hover:opacity-100`}
                  onClick={() => handleAsteroidClick(asteroid.id)}
                  filter="url(#glow)"
                />

                {/* PHA warning pulse */}
                {asteroid.isPHA && (
                  <>
                    <circle
                      cx={asteroid.x}
                      cy={asteroid.y}
                      r={asteroid.size + 3}
                      fill="none"
                      stroke="rgba(239, 68, 68, 0.4)"
                      strokeWidth="2"
                      className="animate-ping"
                    />
                    <circle
                      cx={asteroid.x}
                      cy={asteroid.y}
                      r={asteroid.size + 6}
                      fill="none"
                      stroke="rgba(239, 68, 68, 0.2)"
                      strokeWidth="1"
                      className="animate-pulse"
                    />
                  </>
                )}

                {/* Selection targeting reticle */}
                {selectedAsteroid === asteroid.id && (
                  <g>
                    <circle
                      cx={asteroid.x}
                      cy={asteroid.y}
                      r={asteroid.size + 8}
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth="2"
                      strokeDasharray="4,2"
                      className="animate-pulse"
                      filter="url(#glow)"
                    />
                    {/* Targeting crosshairs */}
                    <line
                      x1={asteroid.x - asteroid.size - 12}
                      y1={asteroid.y}
                      x2={asteroid.x - asteroid.size - 4}
                      y2={asteroid.y}
                      stroke="#fbbf24"
                      strokeWidth="2"
                    />
                    <line
                      x1={asteroid.x + asteroid.size + 4}
                      y1={asteroid.y}
                      x2={asteroid.x + asteroid.size + 12}
                      y2={asteroid.y}
                      stroke="#fbbf24"
                      strokeWidth="2"
                    />
                    <line
                      x1={asteroid.x}
                      y1={asteroid.y - asteroid.size - 12}
                      x2={asteroid.x}
                      y2={asteroid.y - asteroid.size - 4}
                      stroke="#fbbf24"
                      strokeWidth="2"
                    />
                    <line
                      x1={asteroid.x}
                      y1={asteroid.y + asteroid.size + 4}
                      x2={asteroid.x}
                      y2={asteroid.y + asteroid.size + 12}
                      stroke="#fbbf24"
                      strokeWidth="2"
                    />
                  </g>
                )}
              </g>
            ))}
          </svg>
        </div>

        {/* Status panel */}
        <div className="p-3 border-t border-green-500/50 bg-black/80">
          <div className="flex items-center justify-between text-xs font-mono">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                <span className="text-red-400">
                  THREAT ({asteroidPositions.filter((a) => a.isPHA).length})
                </span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 rounded-full bg-green-500"></div>
                <span className="text-green-400">
                  CLEAR ({asteroidPositions.filter((a) => !a.isPHA).length})
                </span>
              </div>
            </div>
            <div className="text-green-400">
              CONTACTS: {asteroidPositions.length}
            </div>
          </div>

          {/* Selected object details */}
          {selectedAsteroid &&
            (() => {
              const selected = asteroidPositions.find(
                (a) => a.id === selectedAsteroid
              );
              return selected ? (
                <div className="mt-2 p-2 bg-green-900/20 rounded border border-green-500/30">
                  <div className="font-mono font-bold text-green-400 text-sm">
                    TARGET: {selected.data.name}
                  </div>
                  <div className="text-xs font-mono text-green-300 mt-1 grid grid-cols-2 gap-2">
                    <div>SIZE: {selected.data.size.toFixed(2)} KM</div>
                    <div>DIST: {selected.data.distance.toFixed(3)} AU</div>
                    <div>VEL: {selected.data.velocity.toFixed(1)} KM/S</div>
                    <div>
                      CLASS:{" "}
                      <span
                        className={
                          selected.isPHA ? "text-red-400" : "text-green-400"
                        }
                      >
                        {selected.isPHA ? "THREAT" : "CLEAR"}
                      </span>
                    </div>
                  </div>
                </div>
              ) : null;
            })()}
        </div>
      </div>
    </div>
  );
};

export default SpaceVisualization;
