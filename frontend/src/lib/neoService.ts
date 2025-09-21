import apiClient from "./apiClient";

interface NEOData {
  id: string;
  name: string;
  size: number; // meters
  distance: number; // million km
  velocity: number; // km/s
  threatLevel: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  timeToClosestApproach: number; // hours
  impactProbability: number; // percentage
}

async function getNEOData(): Promise<NEOData[]> {
  const response = await apiClient.get("http://localhost:8000/data/some");
  console.log("Fetched NEO data:", response.data.predictions);
  return response.data.predictions.map(
    (item) =>
      ({
        id: item.object_id,
        name: item.name,
        size: item.size_km * 1000, // Convert km to meters
        distance: item.distance_km / 1_000_000,
        velocity: item.velocity_kms,
        threatLevel:
          item.predicted_risk_level.toUpperCase() as NEOData["threatLevel"],
        timeToClosestApproach: item.eta_closest
          ? Math.max(
              0,
              (new Date(item.eta_closest).getTime() - new Date().getTime()) /
                (1000 * 60 * 60)
            )
          : -1,
        impactProbability: item.impact_probability,
      } satisfies NEOData)
  );
}

function getMockNEOData(): NEOData[] {
  const objects = [
    {
      name: "2024-XK47",
      size: 0.8,
      baseDistance: 0.02,
    },
    {
      name: "2024-YM12",
      size: 1.2,
      baseDistance: 0.05,
    },
    {
      name: "2024-ZN88",
      size: 0.3,
      baseDistance: 0.15,
    },
    {
      name: "2024-QP23",
      size: 2.1,
      baseDistance: 0.08,
    },
    {
      name: "2024-RD56",
      size: 0.6,
      baseDistance: 0.12,
    },
    {
      name: "2024-ST14",
      size: 1.8,
      baseDistance: 0.03,
    },
    {
      name: "2024-TK99",
      size: 0.4,
      baseDistance: 0.2,
    },
    {
      name: "2024-UV67",
      size: 1.0,
      baseDistance: 0.07,
    },
  ];

  return objects.map((obj, index) => {
    const timeVariation = Math.sin(Date.now() / 10000 + index) * 0.01;
    const distance = obj.baseDistance + timeVariation;
    const velocity = 15 + Math.random() * 25;
    const timeToClosest = (distance * 149.6) / (velocity * 0.0036); // Convert to hours

    let threatLevel: NEOData["threatLevel"] = "LOW";
    let impactProbability = 0;

    if (distance < 0.05) {
      threatLevel = obj.size > 1.5 ? "CRITICAL" : "HIGH";
      impactProbability = obj.size > 1.5 ? 0.15 : 0.05;
    } else if (distance < 0.1) {
      threatLevel = obj.size > 1 ? "HIGH" : "MEDIUM";
      impactProbability = obj.size > 1 ? 0.03 : 0.01;
    }

    return {
      id: `neo-${index}`,
      name: obj.name,
      size: obj.size,
      distance,
      velocity,
      threatLevel,
      timeToClosestApproach: timeToClosest,
      impactProbability,
    };
  });
}

export { getMockNEOData, getNEOData };
export type { NEOData };
