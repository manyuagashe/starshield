import React, { useMemo, useState } from "react";
import { NEOData } from "../lib/neoService";

interface SpaceVisualizationProps {
  neoData: NEOData[];
  className?: string;
  setSelectedNEO?: (neo: string | null) => void;
  selectedNEO?: string | null;
}

interface AsteroidPosition {
  id: string;
  x: number;
  y: number;
  size: number;
  isPHA: boolean;
  data: NEOData;
  closestApproachRadius: number;
  opacity: number;
}

const SpaceVisualization: React.FC<SpaceVisualizationProps> = ({
  neoData,
  setSelectedNEO,
  selectedNEO = null,
  className = "",
}) => {
  const [currentWeek, setCurrentWeek] = useState(0);

  // Visualization dimensions
  const width = 500;
  const height = 500;
  const centerX = width / 2;
  const centerY = height / 2;
  const earthRadius = 20;

  // Calculate time range and weeks from data
  const timeRange = useMemo(() => {
    if (!neoData.length)
      return { weeks: 1, startDate: new Date(), endDate: new Date() };

    const dates = neoData.map((neo) => new Date(neo.etaClosest));
    const startDate = new Date(Math.min(...dates.map((d) => d.getTime())));
    const endDate = new Date(Math.max(...dates.map((d) => d.getTime())));

    // Calculate number of weeks between start and end
    const msPerWeek = 7 * 24 * 60 * 60 * 1000;
    const totalWeeks = Math.max(
      1,
      Math.ceil((endDate.getTime() - startDate.getTime()) / msPerWeek)
    );

    return { weeks: totalWeeks, startDate, endDate };
  }, [neoData]);

  // Calculate asteroids for current week
  const asteroidPositions = useMemo((): AsteroidPosition[] => {
    if (!neoData.length) return [];

    const now = new Date();
    const weekStart = new Date(timeRange.startDate);
    weekStart.setDate(weekStart.getDate() + currentWeek * 7);

    const weekEnd = new Date(weekStart);
    weekEnd.setDate(weekEnd.getDate() + 7);

    // Filter asteroids for this week
    const weekAsteroids = neoData.filter((neo) => {
      const etaDate = new Date(neo.etaClosest);
      return etaDate >= weekStart && etaDate < weekEnd;
    });

    if (!weekAsteroids.length) return [];

    // Find min/max distances for normalization
    const distances = weekAsteroids.map((neo) => neo.distance);
    const minDistance = Math.min(...distances);
    const maxDistance = Math.max(...distances);

    // Find min/max sizes for normalization
    const sizes = weekAsteroids.map((neo) => neo.size);
    const minSize = Math.min(...sizes);
    const maxSize = Math.max(...sizes);

    // Available radius range (closest approach distance from Earth)
    const minRadius = earthRadius + 15; // Increased buffer from Earth (was +10)
    const maxRadius = Math.min(width, height) / 2 - 30; // Maximum distance

    // Collision detection function
    const checkCollision = (
      x: number,
      y: number,
      size: number,
      existingPositions: AsteroidPosition[]
    ): boolean => {
      const minDistance = size * 2; // Minimum distance between asteroid centers
      return existingPositions.some((existing) => {
        const dx = x - existing.x;
        const dy = y - existing.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        return distance < minDistance;
      });
    };

    const positions: AsteroidPosition[] = [];

    weekAsteroids.forEach((neo, index) => {
      // Normalize closest approach distance to radius
      const normalizedDistance =
        (neo.distance - minDistance) / (maxDistance - minDistance);
      const closestApproachRadius =
        minRadius + normalizedDistance * (maxRadius - minRadius);

      // Calculate time relative to week start
      const etaDate = new Date(neo.etaClosest);
      const timeFromWeekStart =
        (etaDate.getTime() - weekStart.getTime()) / (1000 * 60 * 60); // hours

      // Assign each asteroid a unique orbital angle based on its index
      const baseAngle = (index / weekAsteroids.length) * 2 * Math.PI;
      // Add some orbital progression based on time within the week
      const orbitalSpeed = 0.1 + (neo.velocity / 100) * 0.05; // Faster objects orbit faster
      let currentAngle = baseAngle + timeFromWeekStart * orbitalSpeed;

      // Calculate initial position
      let x = centerX + closestApproachRadius * Math.cos(currentAngle);
      let y = centerY + closestApproachRadius * Math.sin(currentAngle);

      // Normalize size
      const normalizedSize =
        minSize === maxSize
          ? 6
          : 3 + ((neo.size - minSize) / (maxSize - minSize)) * 12;

      // Check for collisions and adjust angle if needed
      let attempts = 0;
      const maxAttempts = 36; // Try up to 36 different angles (10 degree increments)
      while (
        (currentAngle === 0 ||
          checkCollision(x, y, normalizedSize, positions)) &&
        attempts < maxAttempts
      ) {
        currentAngle += Math.PI / 18; // 10 degrees
        x = centerX + closestApproachRadius * Math.cos(currentAngle);
        y = centerY + closestApproachRadius * Math.sin(currentAngle);
        attempts++;
      }

      // If still colliding after max attempts, place at a random angle on the orbit
      if (checkCollision(x, y, normalizedSize, positions)) {
        currentAngle = Math.random() * 2 * Math.PI;
        x = centerX + closestApproachRadius * Math.cos(currentAngle);
        y = centerY + closestApproachRadius * Math.sin(currentAngle);
      }

      // Final safety check: ensure minimum distance from Earth center
      const distanceFromCenter = Math.sqrt(
        (x - centerX) ** 2 + (y - centerY) ** 2
      );
      const minDistanceFromEarth = earthRadius + normalizedSize + 5; // Earth radius + asteroid size + buffer
      if (distanceFromCenter < minDistanceFromEarth) {
        // Adjust position to maintain minimum distance
        const angle = Math.atan2(y - centerY, x - centerX);
        x = centerX + minDistanceFromEarth * Math.cos(angle);
        y = centerY + minDistanceFromEarth * Math.sin(angle);
      }

      // Calculate opacity based on how close the object is to its closest approach time
      const timeProximity = Math.abs(timeFromWeekStart - 84) / 84; // 84 hours = 3.5 days (middle of week)
      const opacity = Math.max(0.3, Math.min(1, 1 - timeProximity * 0.7));

      positions.push({
        id: neo.id,
        x,
        y,
        size: normalizedSize,
        isPHA: neo.isPHA,
        data: neo,
        closestApproachRadius,
        opacity,
      });
    });

    return positions;
  }, [
    neoData,
    currentWeek,
    timeRange,
    width,
    height,
    centerX,
    centerY,
    earthRadius,
  ]);

  const handleAsteroidClick = (asteroidId: string) => {
    setSelectedNEO(selectedNEO === asteroidId ? null : asteroidId);
  };

  const handleWeekChange = (week: number) => {
    setCurrentWeek(Math.max(0, Math.min(timeRange.weeks - 1, week)));
  };

  return (
    <div className={`relative ${className}`}>
      <div className="command-panel overflow-hidden relative p-2 pt-1">
        {/* Header */}
        <div className="p-3 border-b border-green-500/50 ">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-lg font-bold glow-primary tracking-wider mb-1">
                RADAR SCOPE
              </h3>
              <p className="text-xs text-muted-foreground">
                TACTICAL DISPLAY - NEO TRACKING
              </p>
            </div>
            <div className="text-xs font-mono text-green-400">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                <span className="glow-primary">ACTIVE</span>
              </div>
            </div>
          </div>
        </div>

        {/* Radar Display */}
        <div className="p-4 relative flex justify-center">
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
            className="rounded-lg bg-card relative"
            style={{
              // filter: "drop-shadow(0 0 10px rgba(34, 197, 94, 0.3))",
              background:
                "radial-gradient(circle at center, #001a00 0%, hsl(var(--card)) 70%)",
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
            {[90, 180, 270, 0].map((angle, i) => (
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

            {/* Orbital trails */}
            {asteroidPositions.map((asteroid, i) => {
              const trailPoints = [];
              for (let i = 0; i <= 20; i++) {
                const angle = (i / 20) * 2 * Math.PI;
                const x =
                  centerX + asteroid.closestApproachRadius * Math.cos(angle);
                const y =
                  centerY + asteroid.closestApproachRadius * Math.sin(angle);
                trailPoints.push(`${x},${y}`);
              }
              return (
                <circle
                  key={`trail-${asteroid.id}`}
                  cx={centerX}
                  cy={centerY}
                  r={asteroid.closestApproachRadius}
                  fill="none"
                  stroke="rgba(34, 197, 94, 0.2)"
                  strokeWidth="1"
                  strokeDasharray="3,3"
                />
              );
            })}

            {/* Asteroids (radar blips) */}
            {asteroidPositions.map((asteroid) => (
              <g key={asteroid.id}>
                {/* Trajectory line for selected asteroid */}
                {selectedNEO === asteroid.id &&
                  (() => {
                    // Calculate trajectory line tangent to circle at asteroid's distance
                    const asteroidDistancePixels = Math.sqrt(
                      (asteroid.x - centerX) ** 2 + (asteroid.y - centerY) ** 2
                    );

                    // Angle from Earth center to asteroid
                    const asteroidAngle = Math.atan2(
                      asteroid.y - centerY,
                      asteroid.x - centerX
                    );

                    // Tangent line is perpendicular to radius at asteroid position
                    const tangentAngle = asteroidAngle + Math.PI / 2;

                    // Make sure we extend the line far enough in both directions
                    const lineExtension = Math.max(width, height);
                    const dx = Math.cos(tangentAngle);
                    const dy = Math.sin(tangentAngle);

                    const startX = asteroid.x - lineExtension * dx;
                    const startY = asteroid.y - lineExtension * dy;
                    const endX = asteroid.x + lineExtension * dx;
                    const endY = asteroid.y + lineExtension * dy;

                    return (
                      <line
                        x1={startX}
                        y1={startY}
                        x2={endX}
                        y2={endY}
                        stroke="#fbbf24"
                        strokeWidth="2"
                        strokeDasharray="5,5"
                        opacity="0.7"
                        filter="url(#glow)"
                      />
                    );
                  })()}

                {/* Subtle pulse ring for all asteroids */}
                <circle
                  cx={asteroid.x}
                  cy={asteroid.y}
                  r={asteroid.size + 6}
                  fill="none"
                  stroke={
                    asteroid.isPHA
                      ? "rgba(239, 68, 68, 0.4)"
                      : "rgba(34, 197, 94, 0.4)"
                  }
                  strokeWidth="1.5"
                  className="animate-pulse"
                />

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
                    selectedNEO === asteroid.id ? "opacity-100" : "opacity-80"
                  } hover:opacity-100`}
                  onClick={() => handleAsteroidClick(asteroid.id)}
                  filter="url(#glow)"
                />

                {/* PHA warning pulse */}
                {asteroid.isPHA && (
                  <circle
                    cx={asteroid.x}
                    cy={asteroid.y}
                    r={asteroid.size + 8}
                    fill="none"
                    stroke="rgba(239, 68, 68, 0.2)"
                    strokeWidth="1"
                    className="animate-pulse"
                  />
                )}

                {/* Selection targeting reticle */}
                {selectedNEO === asteroid.id && (
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

        {/* Timeline Controls */}
        <div className="px-4 py-3 border-t border-green-500/30  from-black via-gray-900 to-black">
          <div className="flex items-center justify-between mb-2">
            <div className="text-xs font-mono text-green-400">
              <span className="text-xs text-muted-foreground">
                TEMPORAL NAVIGATION
              </span>
            </div>
            <div className="text-xs text-muted-foreground">
              WEEK {currentWeek + 1} OF {timeRange.weeks}
            </div>
          </div>

          {/* Timeline Slider */}
          <div className="relative">
            {/* Timeline background with grid */}
            <div className="h-8 bg-card border rounded relative overflow-hidden">
              {/* Grid lines */}
              <div className="absolute inset-0 opacity-20">
                {Array.from({ length: timeRange.weeks + 1 }).map((_, i) => (
                  <div
                    key={i}
                    className="absolute top-0 bottom-0 w-px bg-green-500/30"
                    style={{ left: `${(i / timeRange.weeks) * 100}%` }}
                  />
                ))}
              </div>

              {/* Week markers */}
              {Array.from({ length: timeRange.weeks }).map((_, i) => (
                <div
                  key={i}
                  className="absolute top-1 bottom-1 w-1 bg-green-500/20 rounded"
                  style={{
                    left: `${
                      timeRange.weeks > 1
                        ? (i / (timeRange.weeks - 1)) * 100
                        : 50
                    }%`,
                    transform: "translateX(-50%)",
                  }}
                />
              ))}

              {/* Current position indicator */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-green-400 rounded shadow-lg shadow-green-400/50"
                style={{
                  left: `${
                    timeRange.weeks > 1
                      ? (currentWeek / (timeRange.weeks - 1)) * 100
                      : 50
                  }%`,
                  transform:
                    timeRange.weeks - 1 === currentWeek
                      ? "translateX(-100%)"
                      : undefined,
                }}
              />
            </div>

            {/* Slider input */}
            <input
              type="range"
              min="0"
              max={timeRange.weeks - 1}
              value={currentWeek}
              onChange={(e) => handleWeekChange(parseInt(e.target.value))}
              className="absolute inset-0 w-full h-8 opacity-0 cursor-pointer z-10"
            />
          </div>

          {/* Navigation buttons */}
          <div className="flex items-center justify-center gap-4 mt-2">
            <button
              onClick={() => handleWeekChange(currentWeek - 1)}
              disabled={currentWeek === 0}
              className="px-4 py-1 bg-gray-800 border border-green-500/50 text-green-400 font-mono text-sm rounded disabled:opacity-30 disabled:cursor-not-allowed hover:bg-green-900/50 hover:border-green-400 transition-all duration-200"
            >
              ◀ PREVIOUS
            </button>

            <div className="text-xs font-mono text-green-300 px-3 py-1 bg-black/50 border border-green-500/30 rounded">
              {new Date(
                timeRange.startDate.getTime() +
                  currentWeek * 7 * 24 * 60 * 60 * 1000
              ).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}{" "}
              -{" "}
              {new Date(
                timeRange.startDate.getTime() +
                  (currentWeek + 1) * 7 * 24 * 60 * 60 * 1000
              ).toLocaleDateString("en-US", {
                month: "short",
                day: "numeric",
                year: "numeric",
              })}
            </div>

            <button
              onClick={() => handleWeekChange(currentWeek + 1)}
              disabled={currentWeek === timeRange.weeks - 1}
              className="px-4 py-1 bg-gray-800 border border-green-500/50 text-green-400 font-mono text-sm rounded disabled:opacity-30 disabled:cursor-not-allowed hover:bg-green-900/50 hover:border-green-400 transition-all duration-200"
            >
              NEXT ▶
            </button>
          </div>
        </div>

        {/* Status panel */}
        <div className="p-3 border-t border-green-500/50">
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
          {selectedNEO &&
            (() => {
              const selected = asteroidPositions.find(
                (a) => a.id === selectedNEO
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
