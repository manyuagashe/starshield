import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";
import { 
  HelpCircle, 
  ChevronLeft, 
  ChevronRight, 
  Shield, 
  Satellite, 
  Calculator, 
  BarChart3,
  Zap,
  Globe,
  Target,
  Brain,
  CheckCircle,
  Play,
  Pause,
  Sparkles,
  Activity,
  Radio,
  ExternalLink,
  AlertTriangle,
  TrendingUp,
  Monitor
} from "lucide-react";
import { useState, useEffect } from "react";
import { getNEOData } from "../lib/neoService";
import "./HowItWorks.css";

interface TutorialStep {
  id: number;
  title: string;
  icon: React.ReactNode;
  content: React.ReactNode;
  interactive?: boolean;
}

export const HowItWorks = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [simulationStep, setSimulationStep] = useState(0);
  const [isOpen, setIsOpen] = useState(false);
  const [konami, setKonami] = useState([]);
  const [neoData, setNeoData] = useState([]);
  const konamiCode = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight'];

  // Get NEO data for reactive display
  useEffect(() => {
    setNeoData(getNEOData());
  }, []);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!isOpen) return;
      
      // Konami code detection
      const newKonami = [...konami, event.code].slice(-8);
      setKonami(newKonami);
      
      if (JSON.stringify(newKonami) === JSON.stringify(konamiCode)) {
        // Easter egg activated!
        setKonami([]);
        // Add some fun effect here
      }
      
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          prevStep();
          break;
        case 'ArrowRight':
          event.preventDefault();
          nextStep();
          break;
        case 'Escape':
          setIsOpen(false);
          break;
        case ' ':
          if (currentStep === 2) {
            event.preventDefault();
            setIsPlaying(!isPlaying);
          }
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, currentStep, isPlaying, konami]);

  // Auto-advance simulation
  useEffect(() => {
    if (isPlaying && currentStep === 2) { // Risk calculation step
      const interval = setInterval(() => {
        setSimulationStep((prev) => {
          const next = prev + 1;
          if (next >= 4) {
            setIsPlaying(false); // Stop after completing one cycle
            return 3; // Stay at final step
          }
          return next;
        });
      }, 1000);
      return () => clearInterval(interval);
    }
  }, [isPlaying, currentStep]);

  const steps: TutorialStep[] = [
    {
      id: 0,
      title: "Welcome to STAR SHIELD",
      icon: <Shield className="h-6 w-6 text-primary tutorial-float" />,
      content: (
        <div className="space-y-3 w-full max-w-3xl">
          <div className="text-center space-y-2 relative">
            {/* Background scanning effect */}
            <div className="absolute inset-0 opacity-20">
              <div className="w-full h-0.5 bg-primary tutorial-scan-line" />
            </div>
            
            <div className="mx-auto w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center mb-3 tutorial-pulse-glow relative">
              <Shield className="h-8 w-8 text-primary animate-pulse" />
              <div className="absolute inset-0 rounded-full border-2 border-primary/30 animate-ping" />
            </div>
            
            <h3 className="text-lg font-bold glow-primary">Near Earth Object Defense System</h3>
            <p className="text-sm text-muted-foreground max-w-md mx-auto">
              A real-time monitoring system that tracks asteroids and comets passing close to Earth
            </p>
            
            {/* Floating particles effect */}
            <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
              {[...Array(4)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-1 h-1 bg-primary rounded-full tutorial-matrix"
                  style={{
                    left: `${25 + i * 20}%`,
                    animationDelay: `${i * 0.7}s`
                  }}
                />
              ))}
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-3 mt-4">
            <Card className="bg-green-500/10 border-green-500/30 tutorial-card-hover">
              <CardContent className="p-2 text-center">
                <Activity className="h-4 w-4 mx-auto mb-1 text-green-400 animate-pulse" />
                <div className="text-lg font-bold text-green-400">{neoData.length}</div>
                <div className="text-xs text-green-300">Objects Tracked</div>
              </CardContent>
            </Card>
            <Card className="bg-blue-500/10 border-blue-500/30 tutorial-card-hover">
              <CardContent className="p-2 text-center">
                <Radio className="h-4 w-4 mx-auto mb-1 text-blue-400 animate-spin" />
                <div className="text-lg font-bold text-blue-400">
                  {neoData.filter(neo => neo.threatLevel === 'HIGH' || neo.threatLevel === 'CRITICAL').length}
                </div>
                <div className="text-xs text-blue-300">High Priority</div>
              </CardContent>
            </Card>
            <Card className="bg-purple-500/10 border-purple-500/30 tutorial-card-hover">
              <CardContent className="p-2 text-center">
                <Sparkles className="h-4 w-4 mx-auto mb-1 text-purple-400 animate-bounce" />
                <div className="text-lg font-bold text-purple-400">
                  {Math.round(neoData.reduce((sum, neo) => sum + neo.impactProbability, 0) * 1000) / 1000}%
                </div>
                <div className="text-xs text-purple-300">Avg Impact Risk</div>
              </CardContent>
            </Card>
          </div>
        </div>
      )
    },
    {
      id: 1,
      title: "Data Collection Network",
      icon: <Satellite className="h-6 w-6 text-blue-400 animate-pulse" />,
      content: (
        <div className="space-y-3 w-full max-w-3xl">
          <div className="text-center mb-3">
            <p className="text-sm text-muted-foreground mb-2">
              We retrieve real-time asteroid data from NASA's official APIs
            </p>
            <a 
              href="https://api.nasa.gov/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors text-sm underline"
            >
              <Globe className="h-4 w-4" />
              Visit NASA API Portal
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
          
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-lg" />
            <div className="relative p-3 space-y-3">
              
              {/* Connection lines */}
              <div className="absolute left-6 top-12 bottom-12 w-0.5 bg-gradient-to-b from-green-400 via-orange-400 to-blue-400 tutorial-data-flow" />
              
              <div className="flex items-center gap-3 p-2 rounded-lg bg-primary/10 border border-primary/30 hover:bg-primary/20 transition-all cursor-pointer tutorial-card-hover group">
                <div className="relative">
                  <Globe className="h-5 w-5 text-green-400 group-hover:animate-spin" />
                  <div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-green-400 rounded-full animate-ping" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-sm">NASA NEO Web Service (NeoWs)</div>
                  <div className="text-xs text-muted-foreground">RESTful API providing asteroid orbital data, size estimates, and close approach details</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <div className="text-xs text-green-400">● LIVE</div>
                    <div className="text-xs text-muted-foreground">HTTP requests every 15 min</div>
                  </div>
                </div>
                <Badge variant="secondary" className="text-xs">JSON</Badge>
              </div>
              
              <div className="flex items-center gap-3 p-2 rounded-lg bg-primary/10 border border-primary/30 hover:bg-primary/20 transition-all cursor-pointer tutorial-card-hover group">
                <div className="relative">
                  <Target className="h-5 w-5 text-orange-400 group-hover:animate-bounce" />
                  <div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-orange-400 rounded-full animate-ping" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-sm">JPL SENTRY API</div>
                  <div className="text-xs text-muted-foreground">Impact probability calculations and risk assessments for potentially hazardous asteroids</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <div className="text-xs text-orange-400">● COMPUTING</div>
                    <div className="text-xs text-muted-foreground">Processed via API calls</div>
                  </div>
                </div>
                <Badge variant="secondary" className="text-xs">API</Badge>
              </div>
              
              <div className="flex items-center gap-3 p-2 rounded-lg bg-primary/10 border border-primary/30 hover:bg-primary/20 transition-all cursor-pointer tutorial-card-hover group">
                <div className="relative">
                  <BarChart3 className="h-5 w-5 text-blue-400 group-hover:animate-pulse" />
                  <div className="absolute -top-0.5 -right-0.5 w-1.5 h-1.5 bg-blue-400 rounded-full animate-ping" />
                </div>
                <div className="flex-1">
                  <div className="font-semibold text-sm">CNEOS Close Approach Data</div>
                  <div className="text-xs text-muted-foreground">Trajectory predictions and orbital mechanics from NASA's Center for NEO Studies</div>
                  <div className="flex items-center gap-2 mt-0.5">
                    <div className="text-xs text-blue-400">● ANALYZING</div>
                    <div className="text-xs text-muted-foreground">Fetched via web scraping</div>
                  </div>
                </div>
                <Badge variant="secondary" className="text-xs">CSV</Badge>
              </div>
            </div>
          </div>
          
          {/* Data flow visualization */}
          <div className="mt-3 p-3 bg-muted/20 rounded-lg border border-muted">
            <div className="text-center text-xs text-muted-foreground mb-2">Our Data Pipeline</div>
            <div className="flex items-center justify-between text-xs">
              <div>API Fetch</div>
              <div className="flex-1 mx-2 h-0.5 bg-gradient-to-r from-green-400 to-blue-400 tutorial-progress"></div>
              <div>Parse JSON</div>
              <div className="flex-1 mx-2 h-0.5 bg-gradient-to-r from-blue-400 to-purple-400 tutorial-progress"></div>
              <div>ML Analysis</div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 2,
      title: "ML Risk Assessment",
      icon: <Brain className="h-6 w-6 text-orange-400" />,
      content: (
        <div className="space-y-4 w-full max-w-4xl">
          {/* Main Explanation - Top Focus */}
          <div className="text-center space-y-3 p-4 bg-gradient-to-r from-orange-500/10 to-purple-500/10 rounded-lg border border-orange-500/30">
            <div className="flex items-center justify-center gap-2">
              <Brain className="h-6 w-6 text-orange-400" />
              <span className="font-bold text-lg">Random Forest Algorithm</span>
            </div>
            <p className="text-sm text-muted-foreground max-w-2xl mx-auto leading-relaxed">
              Our model uses an ensemble of 100+ decision trees that each analyze asteroid characteristics independently. 
              Like a panel of experts, each tree "votes" on the threat level, and the majority decision becomes our final classification.
            </p>
            <div className="flex justify-center gap-4 text-xs">
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                <span>Size Analysis</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span>Velocity Check</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
                <span>Distance Eval</span>
              </div>
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                <span>Final Vote</span>
              </div>
            </div>
          </div>

          {/* Control Panel */}
          <div className="flex items-center justify-center gap-3">
            <Button
              size="sm"
              variant={isPlaying ? "destructive" : "default"}
              onClick={() => setIsPlaying(!isPlaying)}
              className="gap-2"
            >
              {isPlaying ? <Pause className="h-3 w-3" /> : <Play className="h-3 w-3" />}
              {isPlaying ? "Pause" : "Watch"} Pipeline
            </Button>
            <Badge variant="outline" className="text-xs">
              Analyzing: {neoData.length > 0 ? neoData[0].name : 'No Data'}
            </Badge>
          </div>
          
          {/* Pipeline Timeline */}
          <div className="relative pb-4">
            {/* Pipeline Flow Line - positioned at icon level */}
            <div className="absolute top-8 left-0 right-0 h-0.5 bg-gradient-to-r from-blue-400 to-orange-400 opacity-30"></div>
            <div 
              className="absolute top-8 left-0 h-0.5 bg-gradient-to-r from-blue-400 to-orange-400 transition-all duration-1000"
              style={{ 
                width: simulationStep >= 3 ? '100%' : simulationStep >= 2 ? '75%' : simulationStep >= 1 ? '50%' : simulationStep >= 0 ? '25%' : '0%'
              }}
            ></div>
            
            {/* Pipeline Steps */}
            <div className="flex justify-between items-start relative z-10">
              {/* Step 1: Size */}
              <div className={`flex flex-col items-center p-3 rounded-lg transition-all duration-500 ${
                simulationStep >= 0 ? 'bg-blue-500/20 border border-blue-500/50 scale-105' : 'bg-muted/10'
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-3 ${
                  simulationStep >= 0 ? 'bg-blue-400 animate-pulse' : 'bg-gray-400'
                }`}>
                  <Target className="h-4 w-4 text-white" />
                </div>
                <span className="text-xs font-medium">Size</span>
                <span className="text-xs text-muted-foreground">
                  {neoData.length > 0 ? `${(neoData[0].size * 1000).toFixed(0)}m` : '---'}
                </span>
                {simulationStep >= 0 && neoData.length > 0 && (
                  <div className="text-xs text-blue-400 mt-1">
                    {neoData[0].size > 1 ? '⚠ Large' : '✓ Small'}
                  </div>
                )}
              </div>

              {/* Step 2: Velocity */}
              <div className={`flex flex-col items-center p-3 rounded-lg transition-all duration-500 ${
                simulationStep >= 1 ? 'bg-green-500/20 border border-green-500/50 scale-105' : 'bg-muted/10'
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-3 ${
                  simulationStep >= 1 ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
                }`}>
                  <Zap className="h-4 w-4 text-white" />
                </div>
                <span className="text-xs font-medium">Velocity</span>
                <span className="text-xs text-muted-foreground">
                  {neoData.length > 0 ? `${neoData[0].velocity.toFixed(1)} km/s` : '---'}
                </span>
                {simulationStep >= 1 && neoData.length > 0 && (
                  <div className="text-xs text-green-400 mt-1">
                    {neoData[0].velocity > 20 ? '⚠ Fast' : '✓ Moderate'}
                  </div>
                )}
              </div>

              {/* Step 3: Distance */}
              <div className={`flex flex-col items-center p-3 rounded-lg transition-all duration-500 ${
                simulationStep >= 2 ? 'bg-purple-500/20 border border-purple-500/50 scale-105' : 'bg-muted/10'
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-3 ${
                  simulationStep >= 2 ? 'bg-purple-400 animate-pulse' : 'bg-gray-400'
                }`}>
                  <Radio className="h-4 w-4 text-white" />
                </div>
                <span className="text-xs font-medium">Distance</span>
                <span className="text-xs text-muted-foreground">
                  {neoData.length > 0 ? `${neoData[0].distance.toFixed(3)} AU` : '---'}
                </span>
                {simulationStep >= 2 && neoData.length > 0 && (
                  <div className="text-xs text-purple-400 mt-1">
                    {neoData[0].distance < 0.05 ? '⚠ Close' : '✓ Safe'}
                  </div>
                )}
              </div>

              {/* Step 4: Final Decision */}
              <div className={`flex flex-col items-center p-3 rounded-lg transition-all duration-500 ${
                simulationStep >= 3 ? 'bg-orange-500/20 border border-orange-500/50 scale-110 shadow-lg' : 'bg-muted/10'
              }`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center mb-3 ${
                  simulationStep >= 3 ? 'bg-orange-400 animate-bounce' : 'bg-gray-400'
                }`}>
                  <Brain className="h-4 w-4 text-white" />
                </div>
                <span className="text-xs font-medium">Decision</span>
                <span className={`text-xs font-bold ${simulationStep >= 3 ? 'text-orange-400' : 'text-muted-foreground'}`}>
                  {simulationStep >= 3 && neoData.length > 0 ? neoData[0].threatLevel : '---'}
                </span>
                {simulationStep >= 3 && neoData.length > 0 && (
                  <div className="text-xs text-orange-400 mt-1">
                    {neoData[0].impactProbability.toFixed(2)}% risk
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Bottom Summary */}
          {simulationStep >= 3 && neoData.length > 0 && (
            <div className="text-center p-3 bg-gradient-to-r from-orange-500/10 to-red-500/10 rounded-lg border border-orange-500/30 animate-fade-in">
              <div className="flex items-center justify-center gap-2 mb-1">
                <CheckCircle className="h-4 w-4 text-orange-400" />
                <span className="text-sm font-medium">Analysis Complete</span>
              </div>
              <p className="text-xs text-muted-foreground">
                ML classified {neoData[0].name} as {neoData[0].threatLevel} risk • 
                Impact probability: {neoData[0].impactProbability.toFixed(3)}%
              </p>
            </div>
          )}
        </div>
      ),
      interactive: true
    },
    {
      id: 3,
      title: "Dashboard Overview",
      icon: <BarChart3 className="h-6 w-6 text-green-400 animate-bounce" />,
      content: (
        <div className="space-y-3 w-full max-w-3xl">
          {/* Threat Level Key */}
          <div className="bg-gradient-to-r from-green-500/10 to-red-500/10 p-3 rounded-lg border border-primary/30">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle className="h-4 w-4 text-orange-400" />
              <div className="font-semibold text-sm">Current Threat Distribution</div>
            </div>
            <div className="grid grid-cols-4 gap-3">
              <div className="flex flex-col items-center gap-2">
                <div className="w-4 h-4 rounded bg-green-400"></div>
                <span className="text-xs font-medium">LOW</span>
                <span className="text-lg font-bold text-green-400">
                  {neoData.filter(neo => neo.threatLevel === 'LOW').length}
                </span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <div className="w-4 h-4 rounded bg-yellow-400"></div>
                <span className="text-xs font-medium">MEDIUM</span>
                <span className="text-lg font-bold text-yellow-400">
                  {neoData.filter(neo => neo.threatLevel === 'MEDIUM').length}
                </span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <div className="w-4 h-4 rounded bg-red-400"></div>
                <span className="text-xs font-medium">HIGH</span>
                <span className="text-lg font-bold text-red-400">
                  {neoData.filter(neo => neo.threatLevel === 'HIGH').length}
                </span>
              </div>
              <div className="flex flex-col items-center gap-2">
                <div className="w-4 h-4 rounded bg-red-600"></div>
                <span className="text-xs font-medium">CRITICAL</span>
                <span className="text-lg font-bold text-red-600">
                  {neoData.filter(neo => neo.threatLevel === 'CRITICAL').length}
                </span>
              </div>
            </div>
          </div>

          {/* Object Tracking List */}
          <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 p-3 rounded-lg border border-blue-500/30">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="h-4 w-4 text-blue-400" />
              <div className="font-semibold text-sm">NEO Tracking List</div>
              <Badge className="text-xs bg-blue-500/20 text-blue-400 border-blue-500/30">
                Interactive Table
              </Badge>
            </div>
            <div className="text-xs text-muted-foreground space-y-1">
              <div>• <strong>Purpose:</strong> Comprehensive catalog of all potentially hazardous near-Earth objects</div>
              <div>• <strong>Features:</strong> Search, filter, and sort asteroids by any parameter by clicking on it</div>
              <div>• <strong>Data View:</strong> Size, velocity, distance, threat level, and approach time</div>
            </div>
          </div>

          {/* System Status */}
          <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 p-3 rounded-lg border border-purple-500/30">
            <div className="flex items-center gap-2 mb-2">
              <Monitor className="h-4 w-4 text-purple-400" />
              <div className="font-semibold text-sm">System Status Panel</div>
              <Activity className="h-3 w-3 text-purple-400 animate-pulse" />
            </div>
            <div className="text-xs text-muted-foreground space-y-1">
              <div>• <strong>Purpose:</strong> Real-time overview of planetary defense system health</div>
              <div>• <strong>Monitoring:</strong> API Connection Status</div>
              <div>• <strong>Assessment:</strong> Overall threat environment classification and trends</div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 4,
      title: "Tutorial Complete",
      icon: <CheckCircle className="h-6 w-6 text-green-400 animate-bounce" />,
      content: (
        <div className="space-y-3 w-full max-w-3xl">
          {/* Interactive completion section */}
          <div className="text-center p-3 bg-gradient-to-br from-primary/10 via-green-500/10 to-blue-500/10 rounded-lg border border-primary/30 relative overflow-hidden">
            {/* Animated background */}
            <div className="absolute inset-0 opacity-20">
              {[...Array(6)].map((_, i) => (
                <div
                  key={i}
                  className="absolute w-1 h-1 bg-primary rounded-full tutorial-matrix"
                  style={{
                    left: `${15 + i * 15}%`,
                    top: `${30 + (i % 2) * 30}%`,
                    animationDelay: `${i * 0.5}s`
                  }}
                />
              ))}
            </div>
            
            <div className="relative z-10 space-y-2">
              <div className="relative inline-block">
                <CheckCircle className="h-8 w-8 text-green-400 animate-bounce" />
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-10 h-10 border-2 border-green-400 rounded-full animate-ping opacity-20" />
                </div>
              </div>
              
              <div className="font-semibold">Tutorial Complete! 🎉</div>
              <div className="text-xs text-muted-foreground mb-2">
                You're now ready to explore STAR SHIELD's capabilities
              </div>
            </div>
          </div>
        </div>
      )
    }
  ];

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      setSimulationStep(0);
      setIsPlaying(false);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setSimulationStep(0);
      setIsPlaying(false);
    }
  };

  const resetTutorial = () => {
    setCurrentStep(0);
    setSimulationStep(0);
    setIsPlaying(false);
    setKonami([]);
  };

  return (
    <Dialog open={isOpen} onOpenChange={(open) => {
      setIsOpen(open);
      if (!open) resetTutorial();
    }}>
      <DialogTrigger asChild>
        <Button 
          variant="outline" 
          size="sm" 
          className="gap-2 hover:scale-105 transition-transform relative group"
          onClick={() => setIsOpen(true)}
        >
          <HelpCircle className="h-4 w-4 group-hover:animate-spin" />
          How it works
          <div className="absolute -top-1 -right-1 w-2 h-2 bg-primary rounded-full animate-ping opacity-75" />
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-4xl h-[85vh] overflow-hidden flex flex-col">
        <div className="flex-1 flex flex-col space-y-4 min-h-0">
          {/* Keyboard hints */}
          <div className="absolute top-2 right-16 text-xs text-muted-foreground opacity-60">
            ← → to navigate • ESC to close {currentStep === 2 && '• SPACE to play/pause'}
          </div>
          
          {/* Header with progress */}
          <div className="space-y-3 flex-shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {steps[currentStep].icon}
                <div>
                  <h2 className="text-lg font-bold glow-primary">{steps[currentStep].title}</h2>
                  <p className="text-xs text-muted-foreground">Step {currentStep + 1} of {steps.length}</p>
                </div>
              </div>
              <Badge variant="secondary" className="animate-pulse text-xs">
                Interactive Tutorial
              </Badge>
            </div>
            
            <div className="space-y-2">
              <Progress value={((currentStep + 1) / steps.length) * 100} className="h-1.5" />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>Progress</span>
                <span>{Math.round(((currentStep + 1) / steps.length) * 100)}%</span>
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 min-h-0 flex items-center justify-center">
            <div className="w-full h-full flex items-center justify-center">
              {steps[currentStep].content}
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between pt-3 border-t border-primary/30 bg-background/95 backdrop-blur-sm flex-shrink-0">
            <Button
              variant="outline"
              onClick={prevStep}
              disabled={currentStep === 0}
              className="gap-2 w-24"
            >
              <ChevronLeft className="h-4 w-4" />
              Previous
            </Button>
            
            <div className="flex gap-2">
              {[0, 1, 2, 3, 4].map((index) => (
                <button
                  key={index}
                  onClick={() => {
                    setCurrentStep(index);
                    setSimulationStep(0);
                    setIsPlaying(false);
                  }}
                  className={`w-3 h-3 rounded-full transition-all ${
                    index === currentStep
                      ? 'bg-primary scale-125'
                      : index < currentStep
                      ? 'bg-green-400'
                      : 'bg-muted'
                  }`}
                />
              ))}
            </div>
            
            {currentStep === steps.length - 1 ? (
              <Button
                onClick={() => setIsOpen(false)}
                className="gap-2 bg-green-600 hover:bg-green-700 w-24"
              >
                <CheckCircle className="h-4 w-4" />
                Close
              </Button>
            ) : (
              <Button
                onClick={nextStep}
                disabled={currentStep === steps.length - 1}
                className="gap-2 w-24"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};