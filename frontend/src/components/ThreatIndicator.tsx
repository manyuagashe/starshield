interface ThreatIndicatorProps {
  level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  label?: string;
  className?: string;
}

export const ThreatIndicator = ({ level, label, className = "" }: ThreatIndicatorProps) => {
  const getIndicatorClasses = () => {
    switch (level) {
      case 'LOW':
        return 'threat-low';
      case 'MEDIUM':
        return 'threat-medium';
      case 'HIGH':
        return 'threat-high';
      case 'CRITICAL':
        return 'threat-high critical-pulse';
      default:
        return 'threat-low';
    }
  };

  const getTextColor = () => {
    switch (level) {
      case 'LOW':
        return 'text-success';
      case 'MEDIUM':
        return 'text-warning';
      case 'HIGH':
        return 'text-destructive';
      case 'CRITICAL':
        return 'text-destructive glow-critical';
      default:
        return 'text-success';
    }
  };

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <div className={`threat-indicator ${getIndicatorClasses()}`} />
      {label && (
        <span className={`text-xs font-semibold ${getTextColor()}`}>
          {label}
        </span>
      )}
      <span className={`text-xs font-mono ${getTextColor()}`}>
        {level}
      </span>
    </div>
  );
};