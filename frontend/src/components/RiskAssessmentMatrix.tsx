import { NEOData } from "../lib/neoService";

interface RiskAssessmentMatrixProps {
  neoData: NEOData[];
  loading: boolean;
  onFilterSelect?: (distance: string, velocity: string) => void;
  activeFilters?: { distance: string | null; velocity: string | null };
}

interface MatrixCell {
  count: number;
  phaCount: number;
  phaPercentage: number;
  distanceCategory: string;
  velocityCategory: string;
}

export const RiskAssessmentMatrix = ({
  neoData,
  loading,
  onFilterSelect,
  activeFilters,
}: RiskAssessmentMatrixProps) => {
  // Define distance and velocity categories
  const distanceCategories = [
    { label: "Far", min: 0.04, max: Infinity, description: "> 0.04 AU" },
    { label: "Medium", min: 0.02, max: 0.04, description: "0.02 - 0.04 AU" },
    { label: "Close", min: 0, max: 0.02, description: "< 0.02 AU" },
  ];

  const velocityCategories = [
    { label: "Slow", min: 0, max: 15, description: "< 15 km/s" },
    { label: "Medium", min: 15, max: 25, description: "15 - 25 km/s" },
    { label: "Fast", min: 25, max: Infinity, description: "> 25 km/s" },
  ];

  // Calculate matrix data
  const calculateMatrix = (): MatrixCell[][] => {
    const matrix: MatrixCell[][] = [];

    for (let i = 0; i < distanceCategories.length; i++) {
      matrix[i] = [];
      for (let j = 0; j < velocityCategories.length; j++) {
        const distanceCategory = distanceCategories[i];
        const velocityCategory = velocityCategories[j];

        const cellData = neoData.filter(
          (neo) =>
            neo.distanceAU >= distanceCategory.min &&
            neo.distanceAU < distanceCategory.max &&
            neo.velocity >= velocityCategory.min &&
            neo.velocity < velocityCategory.max
        );

        const phaCount = cellData.filter((neo) => neo.isPHA).length;
        const total = cellData.length;
        const phaPercentage = total > 0 ? (phaCount / total) * 100 : 0;

        matrix[i][j] = {
          count: total,
          phaCount,
          phaPercentage,
          distanceCategory: distanceCategory.label,
          velocityCategory: velocityCategory.label,
        };
      }
    }

    return matrix;
  };

  // Get heat map color based on PHA percentage
  const getHeatMapColor = (percentage: number) => {
    if (percentage === 0)
      return "bg-gradient-to-br from-slate-900/80 to-slate-800/60 border-slate-700/50 text-slate-300";
    if (percentage < 20)
      return "bg-gradient-to-br from-green-900/60 to-emerald-800/40 border-green-600/30 text-green-200";
    if (percentage < 40)
      return "bg-gradient-to-br from-yellow-900/60 to-amber-800/40 border-yellow-600/30 text-yellow-200";
    if (percentage < 60)
      return "bg-gradient-to-br from-orange-900/60 to-orange-800/40 border-orange-600/30 text-orange-200";
    if (percentage < 80)
      return "bg-gradient-to-br from-red-900/60 to-red-800/40 border-red-600/30 text-red-200";
    return "bg-gradient-to-br from-red-800/80 to-red-700/60 border-red-500/40 text-red-100";
  };

  const matrix = calculateMatrix();

  if (loading) {
    return (
      <div className="command-panel relative z-10 bg-background">
        <div className="flex items-center justify-center h-32">
          <div className="w-6 h-6 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
          <span className="ml-2 text-sm text-muted-foreground">
            Loading matrix data...
          </span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="command-panel relative bg-background border border-primary/20"
      style={{ zIndex: 50 }}
    >
      <div className="space-y-4">
        {/* Header */}
        <div className="text-center space-y-2">
          <div className="flex items-center justify-center gap-2">
            <div className="w-6 h-6 rounded-full bg-gradient-to-r from-orange-500 to-red-500 flex items-center justify-center shadow-lg">
              <div className="w-3 h-3 bg-white/90 rounded-sm transform rotate-45"></div>
            </div>
            <h3 className="text-sm font-bold text-primary/80 tracking-wider">
              THREAT ASSESSMENT GRID
            </h3>
          </div>
        </div>

        {/* Spacer */}
        <div className="h-3"></div>

        {/* Radar-style Heat Map */}
        <div className="relative bg-gradient-to-br from-background/50 to-muted/30 rounded-lg p-4 backdrop-blur-sm">
          {/* Velocity labels */}
          <div className="flex items-center gap-2 mb-3">
            {/* Spacer to align with distance labels */}
            <div className="w-16"></div>

            {/* Velocity label columns */}
            <div className="flex-1 flex gap-1">
              {velocityCategories.map((category, index) => (
                <div
                  key={index}
                  className="flex-1 text-xs text-center text-muted-foreground font-mono"
                >
                  <div className="font-semibold text-primary/70">
                    {category.label.toUpperCase()}
                  </div>
                  <div className="text-xs opacity-60">
                    {category.description}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Heat map grid */}
          <div className="space-y-2">
            {distanceCategories.map((distanceCategory, rowIndex) => (
              <div key={rowIndex} className="flex items-center gap-2">
                {/* Distance label */}
                <div className="w-16 text-right text-xs font-mono">
                  <div className="font-semibold text-primary/70">
                    {distanceCategory.label.toUpperCase()}
                  </div>
                  <div className="text-xs text-muted-foreground opacity-60">
                    {distanceCategory.description}
                  </div>
                </div>

                {/* Heat map cells */}
                <div className="flex-1 flex gap-1">
                  {matrix[rowIndex].map((cell, colIndex) => {
                    const intensity = cell.phaPercentage / 100;
                    const isActive =
                      activeFilters?.distance === cell.distanceCategory &&
                      activeFilters?.velocity === cell.velocityCategory;

                    return (
                      <div
                        key={colIndex}
                        className={`flex-1 aspect-square rounded-lg transition-all duration-500 cursor-pointer group relative overflow-hidden ${getHeatMapColor(
                          cell.phaPercentage
                        )} ${
                          isActive
                            ? "ring-2 ring-primary ring-offset-2 ring-offset-background"
                            : ""
                        }`}
                        style={{
                          boxShadow: `0 0 ${intensity * 20}px ${
                            intensity > 0.6
                              ? "rgba(239, 68, 68, 0.6)"
                              : intensity > 0.4
                              ? "rgba(245, 158, 11, 0.5)"
                              : intensity > 0.2
                              ? "rgba(251, 191, 36, 0.4)"
                              : "rgba(16, 185, 129, 0.3)"
                          }`,
                          transform: `scale(${
                            0.95 + intensity * 0.1 + (isActive ? 0.05 : 0)
                          })`,
                        }}
                        title={`Click to filter: ${cell.distanceCategory} & ${
                          cell.velocityCategory
                        } (${cell.count} objects, ${
                          cell.phaCount
                        } PHA - ${cell.phaPercentage.toFixed(1)}%)`}
                        onClick={() =>
                          onFilterSelect?.(
                            cell.distanceCategory,
                            cell.velocityCategory
                          )
                        }
                      >
                        {/* Glowing border for high risk */}
                        {cell.phaPercentage > 40 && (
                          <div className="absolute inset-0 rounded-lg border border-current opacity-50 animate-pulse"></div>
                        )}

                        {/* Hover glow */}
                        <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-lg"></div>

                        {/* Content */}
                        <div className="relative z-10 h-full flex flex-col items-center justify-center p-2 text-center">
                          <div className="text-lg font-bold leading-none mb-1">
                            {cell.count}
                          </div>
                          <div className="text-xs font-semibold opacity-90 mb-1">
                            {cell.phaCount} PHA
                          </div>
                          <div className="text-xs font-bold px-1 py-0.5 rounded bg-black/30 backdrop-blur-sm">
                            {cell.phaPercentage.toFixed(0)}%
                          </div>
                        </div>

                        {/* Risk pulse for critical zones */}
                        {cell.phaPercentage > 60 && (
                          <div className="absolute top-1 right-1 w-2 h-2 bg-red-400 rounded-full">
                            <div className="absolute inset-0 bg-red-400 rounded-full animate-ping opacity-75"></div>
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Compact legend */}
        <div className="flex items-center justify-center gap-3 text-xs bg-muted/20 rounded-lg p-2 border border-primary/10">
          <span className="text-muted-foreground font-mono">THREAT LEVEL:</span>
          {[
            {
              color: "bg-green-800/60 border-green-600/50",
              label: "LOW",
              range: "0-20%",
            },
            {
              color: "bg-yellow-800/60 border-yellow-600/50",
              label: "MED",
              range: "20-40%",
            },
            {
              color: "bg-orange-800/60 border-orange-600/50",
              label: "HIGH",
              range: "40-60%",
            },
            {
              color: "bg-red-800/60 border-red-600/50",
              label: "CRIT",
              range: "60%+",
            },
          ].map((item, index) => (
            <div key={index} className="flex items-center gap-1">
              <div
                className={`w-2 h-2 ${item.color} rounded-full border`}
              ></div>
              <span className="font-mono font-semibold text-primary/70">
                {item.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
