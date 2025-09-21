import apiClient from "./apiClient";

interface NEOData {
  id: string;
  name: string;
  size: number; // meters
  distance: number; // million km
  velocity: number; // km/s
  isPHA: boolean; // is potentially hazardous asteroid
  etaClosest: string; // ISO date string for closest approach
  timeToClosestApproach: number; // hours (derived from etaClosest)
  impactProbability: number; // percentage
}

async function getNEOData(): Promise<NEOData[]> {
  const response = await apiClient.get("/data/all");
  console.log("Fetched NEO data:", response.data.predictions);
  return response.data.predictions.map(
    (item) =>
      ({
        id: `${item.object_id}-${Math.random()}`,
        name: item.name,
        size: item.size_km * 1000, // Convert km to meters
        distance: item.distance_km / 1_000_000,
        velocity: item.velocity_kms,
        isPHA: Boolean(item.input_features.is_pha),
        etaClosest: item.eta_closest,
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
    // Additional non-PHA objects (safe asteroids)
    {
      name: "2024-AB11",
      size: 0.08, // Too small for PHA
      baseDistance: 0.03,
    },
    {
      name: "2024-BC22",
      size: 0.12, // Too small for PHA
      baseDistance: 0.04,
    },
    {
      name: "2024-CD33",
      size: 0.25,
      baseDistance: 0.18, // Too distant for PHA
    },
    {
      name: "2024-DE44",
      size: 0.35,
      baseDistance: 0.22, // Too distant for PHA
    },
    {
      name: "2024-EF55",
      size: 0.09, // Too small for PHA
      baseDistance: 0.08,
    },
    {
      name: "2024-FG66",
      size: 0.45,
      baseDistance: 0.15, // Too distant for PHA
    },
    {
      name: "2024-GH77",
      size: 0.11, // Too small for PHA
      baseDistance: 0.06,
    },
    {
      name: "2024-HI88",
      size: 0.28,
      baseDistance: 0.25, // Too distant for PHA
    },
    {
      name: "2024-IJ99",
      size: 0.13, // Too small for PHA
      baseDistance: 0.09,
    },
    {
      name: "2024-JK00",
      size: 0.52,
      baseDistance: 0.19, // Too distant for PHA
    },
    {
      name: "2024-KL11",
      size: 0.07, // Too small for PHA
      baseDistance: 0.04,
    },
    {
      name: "2024-LM22",
      size: 0.33,
      baseDistance: 0.16, // Too distant for PHA
    },
    {
      name: "2024-MN33",
      size: 0.1, // Too small for PHA
      baseDistance: 0.07,
    },
    {
      name: "2024-NO44",
      size: 0.41,
      baseDistance: 0.14, // Too distant for PHA
    },
    {
      name: "2024-OP55",
      size: 0.06, // Too small for PHA
      baseDistance: 0.05,
    },
    {
      name: "2024-PQ66",
      size: 0.38,
      baseDistance: 0.21, // Too distant for PHA
    },
    {
      name: "2024-QR77",
      size: 0.12, // Too small for PHA
      baseDistance: 0.08,
    },
    {
      name: "2024-RS88",
      size: 0.29,
      baseDistance: 0.17, // Too distant for PHA
    },
    {
      name: "2024-ST99",
      size: 0.08, // Too small for PHA
      baseDistance: 0.06,
    },
    {
      name: "2024-TU00",
      size: 0.46,
      baseDistance: 0.13, // Too distant for PHA
    },
  ];

  // Uniformly distribute etaClosest dates over the next 4 weeks
  const now = Date.now();
  const fourWeeksMs = 28 * 24 * 60 * 60 * 1000;
  return objects.map((obj, index) => {
    const percent = index / objects.length;
    const etaClosestDate = new Date(now + percent * fourWeeksMs);
    const etaClosest = etaClosestDate.toISOString();

    // Calculate timeToClosestApproach in hours
    const timeToClosestApproach =
      (etaClosestDate.getTime() - now) / (1000 * 60 * 60);

    // Simulate distance and velocity
    const timeVariation = Math.sin(Date.now() / 10000 + index) * 0.01;
    const distance = obj.baseDistance + timeVariation;
    const velocity = 15 + Math.random() * 25;

    // Determine if asteroid is potentially hazardous
    const isPHA = distance < 0.05 && obj.size > 0.14;

    // Impact probability increases with size and decreases with distance
    let impactProbability = 0;
    if (isPHA) {
      impactProbability = obj.size > 1.5 ? 0.15 : 0.05;
    } else if (distance < 0.1) {
      impactProbability = obj.size > 1 ? 0.03 : 0.01;
    }

    return {
      id: `neo-${index}`,
      name: obj.name,
      size: obj.size,
      distance,
      velocity,
      isPHA,
      etaClosest,
      timeToClosestApproach,
      impactProbability,
    };
  });
}

export { getMockNEOData, getNEOData };
export type { NEOData };
