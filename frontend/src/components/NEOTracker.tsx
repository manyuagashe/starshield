import { NEOData } from "@/lib/neoService";
import { Badge } from "@/components/ui/badge";
import {
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Search,
  SortAsc,
  SortDesc,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { ThreatIndicator } from "./ThreatIndicator";
import { Button } from "./ui/button";
import { Input } from "./ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

type SortField =
  | "name"
  | "size"
  | "distance"
  | "velocity"
  | "timeToClosestApproach"
  | "isPHA";
type SortDirection = "asc" | "desc";
export const NEOTracker = ({
  neoData,
  loading,
  selectedNEO,
  setSelectedNEO,
  sizeFilter,
  velocityFilter,
  onClearFilters,
}: {
  neoData: NEOData[];
  loading: boolean;
  selectedNEO: string | null;
  setSelectedNEO: (id: string | null) => void;
  sizeFilter?: string | null;
  velocityFilter?: string | null;
  onClearFilters?: () => void;
}) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortField, setSortField] = useState<SortField>("isPHA");
  const [sortDirection, setSortDirection] = useState<SortDirection>("desc");
  const [itemsToShow, setItemsToShow] = useState(12);

  const ITEMS_PER_PAGE = 12;

  useEffect(() => {
    console.log("selectedNEO changed:", selectedNEO);
    if (selectedNEO) {
      console.log("setting search");
      setSearchTerm(neoData.find((neo) => neo.id === selectedNEO)?.name || "");
    }
  }, [selectedNEO, neoData]);

  // Reset pagination when filters change
  useEffect(() => {
    setItemsToShow(ITEMS_PER_PAGE);
  }, [searchTerm, sizeFilter, velocityFilter, sortField, sortDirection]);

  const loadMoreItems = () => {
    setItemsToShow(prev => prev + ITEMS_PER_PAGE);
  };

  const filteredAndSortedData = useMemo(() => {
    let filtered = neoData;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter((neo) =>
        neo.name.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply size filter
    if (sizeFilter) {
      filtered = filtered.filter((neo) => {
        if (sizeFilter === "Small") return neo.size < 0.5;
        if (sizeFilter === "Medium") return neo.size >= 0.5 && neo.size < 1.5;
        if (sizeFilter === "Large") return neo.size >= 1.5;
        return true;
      });
    }

    // Apply velocity filter
    if (velocityFilter) {
      filtered = filtered.filter((neo) => {
        if (velocityFilter === "Slow") return neo.velocity < 15;
        if (velocityFilter === "Medium") return neo.velocity >= 15 && neo.velocity < 25;
        if (velocityFilter === "Fast") return neo.velocity >= 25;
        return true;
      });
    }

    // Apply sorting
    return [...filtered].sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (sortField) {
        case "name":
          aValue = a.name;
          bValue = b.name;
          break;
        case "size":
          aValue = a.size;
          bValue = b.size;
          break;
        case "distance":
          aValue = a.distance;
          bValue = b.distance;
          break;
        case "velocity":
          aValue = a.velocity;
          bValue = b.velocity;
          break;
        case "timeToClosestApproach":
          aValue = a.timeToClosestApproach;
          bValue = b.timeToClosestApproach;
          break;
        case "isPHA":
          aValue = a.isPHA ? 1 : 0;
          bValue = b.isPHA ? 1 : 0;
          break;
        default:
          aValue = a.isPHA ? 1 : 0;
          bValue = b.isPHA ? 1 : 0;
      }

      if (typeof aValue === "string" && typeof bValue === "string") {
        const comparison = aValue.localeCompare(bValue);
        return sortDirection === "asc" ? comparison : -comparison;
      } else {
        const comparison = (aValue as number) - (bValue as number);
        return sortDirection === "asc" ? comparison : -comparison;
      }
    });
  }, [neoData, searchTerm, sortField, sortDirection, sizeFilter, velocityFilter]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortDirection("desc");
    }
  };

  const SortButton = ({
    field,
    children,
  }: {
    field: SortField;
    children: React.ReactNode;
  }) => (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => handleSort(field)}
      className="h-auto p-1 font-mono text-xs hover:bg-primary/20 flex items-center gap-1"
    >
      {children}
      {sortField === field &&
        (sortDirection === "asc" ? (
          <ChevronUp className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3" />
        ))}
    </Button>
  );

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
        <div className="text-xs text-muted-foreground">
          {filteredAndSortedData.length > itemsToShow 
            ? `Showing ${itemsToShow} of ${filteredAndSortedData.length} • Total: ${neoData.length}`
            : `${filteredAndSortedData.length} of ${neoData.length} objects`
          }
        </div>
      </div>

      {/* Filters and Sorting */}
      <div className="flex gap-3 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search by name..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-10 bg-background/50 border-primary/30 text-sm"
          />
          {searchTerm && (
            <button
              onClick={() => {
                setSearchTerm("");
                setSelectedNEO(null);
              }}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Clear search"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        <div className="flex items-center gap-2">
          <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
          <Select
            value={sortField}
            onValueChange={(value) => setSortField(value as SortField)}
          >
            <SelectTrigger className="w-40 bg-background/50 border-primary/30 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="isPHA">PHA Status</SelectItem>
              <SelectItem value="size">Size</SelectItem>
              <SelectItem value="distance">Distance</SelectItem>
              <SelectItem value="velocity">Velocity</SelectItem>
              <SelectItem value="timeToClosestApproach">
                Closest Approach
              </SelectItem>
              <SelectItem value="name">Name</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setSortDirection(sortDirection === "asc" ? "desc" : "asc")
            }
            className="gap-2 bg-background/50 border-primary/30 text-sm"
          >
            {sortDirection === "asc" ? (
              <>
                <SortAsc className="h-4 w-4" />
                Low to High
              </>
            ) : (
              <>
                <SortDesc className="h-4 w-4" />
                High to Low
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Active Filter Indicators */}
      {(sizeFilter || velocityFilter) && (
        <div className="flex items-center gap-2 mb-3 p-2 bg-primary/10 rounded-lg border border-primary/20">
          <span className="text-xs text-muted-foreground font-mono">ACTIVE FILTERS:</span>
          {sizeFilter && (
            <Badge variant="outline" className="text-xs bg-blue-500/20 text-blue-400 border-blue-500/30">
              Size: {sizeFilter}
            </Badge>
          )}
          {velocityFilter && (
            <Badge variant="outline" className="text-xs bg-green-500/20 text-green-400 border-green-500/30">
              Velocity: {velocityFilter}
            </Badge>
          )}
          {onClearFilters && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onClearFilters}
              className="h-6 px-2 text-xs text-muted-foreground hover:text-foreground"
            >
              <X className="h-3 w-3 mr-1" />
              Clear
            </Button>
          )}
        </div>
      )}

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
        ) : filteredAndSortedData.length === 0 && neoData.length === 0 ? (
          <div className="neo-tracking-item">
            <div className="flex items-center justify-center p-8">
              <span className="text-muted-foreground">
                No NEO data available
              </span>
            </div>
          </div>
        ) : (
          <>
            {filteredAndSortedData.slice(0, itemsToShow).map((neo) => (
              <div
                key={neo.id}
                className="neo-tracking-item"
                onClick={() => setSelectedNEO(neo.id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <SortButton field="isPHA">
                      <ThreatIndicator isPHA={neo.isPHA} />
                    </SortButton>
                    <SortButton field="name">
                      <span className="font-bold text-primary">{neo.name}</span>
                    </SortButton>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-4 text-xs">
                  <div>
                    <SortButton field="size">
                      <span className="text-muted-foreground">SIZE:</span>
                    </SortButton>
                    <div className="text-foreground font-mono">
                      {neo.size.toFixed(1)} km
                    </div>
                  </div>
                  <div>
                    <SortButton field="distance">
                      <span className="text-muted-foreground">DISTANCE:</span>
                    </SortButton>
                    <div className="text-foreground font-mono">
                      {formatDistance(neo.distance)} Mkm
                    </div>
                  </div>
                  <div>
                    <SortButton field="velocity">
                      <span className="text-muted-foreground">VELOCITY:</span>
                    </SortButton>
                    <div className="text-foreground font-mono">
                      {neo.velocity.toFixed(1)} km/s
                    </div>
                  </div>
                  <div>
                    <SortButton field="timeToClosestApproach">
                      <span className="text-muted-foreground">
                        ETA CLOSEST:
                      </span>
                    </SortButton>
                    <div className="text-foreground font-mono">
                      {formatTime(neo.timeToClosestApproach)}
                    </div>
                  </div>
                </div>
              </div>
            ))}

            {/* Load More Button */}
            {filteredAndSortedData.length > itemsToShow && (
              <div className="flex justify-center pt-3">
                <Button
                  variant="outline"
                  onClick={loadMoreItems}
                  className="gap-2 bg-background/50 border-primary/30 hover:bg-primary/10 text-sm"
                >
                  <span>Show {Math.min(ITEMS_PER_PAGE, filteredAndSortedData.length - itemsToShow)} more</span>
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </div>
            )}

            {/* Show pagination info */}
            {filteredAndSortedData.length > ITEMS_PER_PAGE && (
              <div className="text-center text-xs text-muted-foreground pt-2 border-t border-primary/20 mt-3">
                Showing {Math.min(itemsToShow, filteredAndSortedData.length)} of {filteredAndSortedData.length} objects
              </div>
            )}

            {filteredAndSortedData.length === 0 && neoData.length > 0 && (
              <div className="text-center py-8 text-muted-foreground">
                <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>No objects match your filters</p>
                <p className="text-xs">
                  Try adjusting your search or threat level filter
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
