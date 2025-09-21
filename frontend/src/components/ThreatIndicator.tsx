interface ThreatIndicatorProps {
  isPHA: boolean;
  label?: string;
  className?: string;
}

export const ThreatIndicator = ({
  isPHA,
  label,
  className = "",
}: ThreatIndicatorProps) => {
  const getIndicatorClasses = () => {
    return isPHA ? "threat-high critical-pulse" : "threat-low";
  };

  const getTextColor = () => {
    return isPHA ? "text-destructive glow-critical" : "text-success";
  };

  const getStatusText = () => {
    return isPHA ? "PHA" : "SAFE";
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
        {getStatusText()}
      </span>
    </div>
  );
};
