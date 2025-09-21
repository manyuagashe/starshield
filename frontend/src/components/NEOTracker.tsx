import { NEOData } from "@/lib/neoService";
import { useMemo, useState } from "react";
import { ThreatIndicator } from "./ThreatIndicator";
import { Input } from "./ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./ui/select";
import { Button } from "./ui/button";
import { ChevronUp, ChevronDown, Search, ArrowUpDown, SortAsc, SortDesc } from "lucide-react";

type SortField = 'name' | 'size' | 'distance' | 'velocity' | 'impactProbability' | 'timeToClosestApproach' | 'threatLevel';
type SortDirection = 'asc' | 'desc';

export const NEOTracker = ({ neoData }: { neoData: NEOData[] }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<SortField>('impactProbability');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const filteredAndSortedData = useMemo(() => {
    let filtered = neoData;

    // Apply search filter
    if (searchTerm) {
      filtered = filtered.filter(neo => 
        neo.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        neo.classification.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Apply sorting
    return [...filtered].sort((a, b) => {
      let aValue: number | string;
      let bValue: number | string;

      switch (sortField) {
        case 'name':
          aValue = a.name;
          bValue = b.name;
          break;
        case 'size':
          aValue = a.size;
          bValue = b.size;
          break;
        case 'distance':
          aValue = a.distance;
          bValue = b.distance;
          break;
        case 'velocity':
          aValue = a.velocity;
          bValue = b.velocity;
          break;
        case 'impactProbability':
          aValue = a.impactProbability;
          bValue = b.impactProbability;
          break;
        case 'timeToClosestApproach':
          aValue = a.timeToClosestApproach;
          bValue = b.timeToClosestApproach;
          break;
        case 'threatLevel':
          // Convert threat level to numeric for sorting
          const threatOrder = { 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4 };
          aValue = threatOrder[a.threatLevel as keyof typeof threatOrder] || 0;
          bValue = threatOrder[b.threatLevel as keyof typeof threatOrder] || 0;
          break;
        default:
          aValue = a.impactProbability;
          bValue = b.impactProbability;
      }

      if (typeof aValue === 'string' && typeof bValue === 'string') {
        const comparison = aValue.localeCompare(bValue);
        return sortDirection === 'asc' ? comparison : -comparison;
      } else {
        const comparison = (aValue as number) - (bValue as number);
        return sortDirection === 'asc' ? comparison : -comparison;
      }
    });
  }, [neoData, searchTerm, sortField, sortDirection]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  const SortButton = ({ field, children }: { field: SortField; children: React.ReactNode }) => (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => handleSort(field)}
      className="h-auto p-1 font-mono text-xs hover:bg-primary/20 flex items-center gap-1"
    >
      {children}
      {sortField === field && (
        sortDirection === 'asc' ? 
          <ChevronUp className="h-3 w-3" /> : 
          <ChevronDown className="h-3 w-3" />
      )}
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
          {filteredAndSortedData.length} of {neoData.length} objects
        </div>
      </div>

      {/* Filters and Sorting */}
      <div className="flex gap-3 mb-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search by name or classification..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 bg-background/50 border-primary/30 text-sm"
          />
        </div>
        <div className="flex items-center gap-2">
          <ArrowUpDown className="h-4 w-4 text-muted-foreground" />
          <Select value={sortField} onValueChange={(value) => setSortField(value as SortField)}>
            <SelectTrigger className="w-40 bg-background/50 border-primary/30 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="threatLevel">Threat Level</SelectItem>
              <SelectItem value="impactProbability">Impact Probability</SelectItem>
              <SelectItem value="size">Size</SelectItem>
              <SelectItem value="distance">Distance</SelectItem>
              <SelectItem value="velocity">Velocity</SelectItem>
              <SelectItem value="timeToClosestApproach">Closest Approach</SelectItem>
              <SelectItem value="name">Name</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')}
            className="gap-2 bg-background/50 border-primary/30 text-sm"
          >
            {sortDirection === 'asc' ? (
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

      <div className="space-y-2">
        {filteredAndSortedData.map((neo) => (
          <div key={neo.id} className="neo-tracking-item">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-3">
                <SortButton field="threatLevel">
                  <ThreatIndicator level={neo.threatLevel} />
                </SortButton>
                <SortButton field="name">
                  <span className="font-bold text-primary">{neo.name}</span>
                </SortButton>
                <span className="text-xs text-muted-foreground">
                  [{neo.classification}]
                </span>
              </div>
              <SortButton field="impactProbability">
                <div className="text-xs text-warning">
                  IMPACT PROB: {neo.impactProbability.toFixed(3)}%
                </div>
              </SortButton>
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
                  <span className="text-muted-foreground">ETA CLOSEST:</span>
                </SortButton>
                <div className="text-foreground font-mono">
                  {formatTime(neo.timeToClosestApproach)}
                </div>
              </div>
            </div>
          </div>
        ))}

        {filteredAndSortedData.length === 0 && (
          <div className="text-center py-8 text-muted-foreground">
            <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
            <p>No objects match your filters</p>
            <p className="text-xs">Try adjusting your search or threat level filter</p>
          </div>
        )}
      </div>
    </div>
  );
};
