import React, { useRef, useState } from 'react';
import { Play, Pause, AlertTriangle, ShieldCheck, CheckCircle2, ShieldAlert, Compass, Lightbulb, ExternalLink } from 'lucide-react';

export interface RiskEventItem {
  id: number;
  event_id: number;
  start_time: number;
  end_time: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  risk_score: number;
  ttc?: number;
  closing_speed?: number;
  minimum_distance?: number;
  vehicles: number[];
  risk_reasons: string[];
  decision_source: string;
}

export interface RecommendationItem {
  id: number;
  category: string;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  title: string;
  summary: string;
  evidence: string[];
  recommended_action: string;
  limitations: string;
  source: 'RULE_BASED' | 'AI_ENHANCED';
}

interface VideoPlayerOverlayProps {
  videoUrl: string;
  duration: number;
  riskEvents: RiskEventItem[];
  recommendations?: RecommendationItem[];
  onSelectEvent?: (event: RiskEventItem) => void;
}

export const VideoPlayerOverlay: React.FC<VideoPlayerOverlayProps> = ({
  videoUrl,
  duration,
  riskEvents,
  recommendations = [],
  onSelectEvent
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [selectedEvent, setSelectedEvent] = useState<RiskEventItem | null>(null);
  const [activeTab, setActiveTab] = useState<'RISKS' | 'RECOMMENDATIONS'>('RISKS');

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const seekToTime = (timeInSeconds: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timeInSeconds;
      setCurrentTime(timeInSeconds);
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleEventClick = (ev: RiskEventItem) => {
    setSelectedEvent(ev);
    seekToTime(ev.start_time);
    if (onSelectEvent) onSelectEvent(ev);
  };

  const seekToEventId = (eventIdNum: number) => {
    const ev = riskEvents.find(e => e.event_id === eventIdNum || e.id === eventIdNum);
    if (ev) {
      handleEventClick(ev);
      setActiveTab('RISKS');
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', gap: '20px' }}>
      {/* Video Player & Interactive Timeline Markers */}
      <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <div style={{ position: 'relative', width: '100%', borderRadius: '8px', overflow: 'hidden', background: '#000000' }}>
          <video
            ref={videoRef}
            src={videoUrl}
            onTimeUpdate={handleTimeUpdate}
            style={{ width: '100%', display: 'block', maxHeight: '520px', objectFit: 'contain' }}
            controls={false}
          />
        </div>

        {/* Playback Controls & Time Scrubber */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px', padding: '4px 8px' }}>
          <button
            onClick={togglePlay}
            style={{ background: '#6366f1', border: 'none', color: '#fff', padding: '8px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
          >
            {isPlaying ? <Pause style={{ width: '18px', height: '18px' }} /> : <Play style={{ width: '18px', height: '18px', marginLeft: '2px' }} />}
          </button>

          <div style={{ flex: 1 }}>
            <input
              type="range"
              min={0}
              max={duration || 10}
              step={0.1}
              value={currentTime}
              onChange={(e) => seekToTime(parseFloat(e.target.value))}
              style={{ width: '100%', accentColor: '#6366f1', cursor: 'pointer' }}
            />
          </div>

          <div style={{ fontSize: '0.85rem', color: '#94a3b8', fontFamily: 'monospace' }}>
            {currentTime.toFixed(1)}s / {(duration || 10).toFixed(1)}s
          </div>
        </div>

        {/* Interactive Risk Event Timeline Markers */}
        <div style={{ marginTop: '8px', borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '12px' }}>
          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8', marginBottom: '8px' }}>
            INTERACTIVE RISK TIMELINE MARKERS
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {riskEvents.length === 0 ? (
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>No risk events recorded in timeline.</span>
            ) : (
              riskEvents.map((ev) => {
                const badgeClass = ev.risk_level === 'HIGH' ? 'badge-high' : ev.risk_level === 'MEDIUM' ? 'badge-medium' : 'badge-low';
                const isSelected = selectedEvent?.id === ev.id;
                return (
                  <button
                    key={ev.id}
                    onClick={() => handleEventClick(ev)}
                    className={badgeClass}
                    style={{
                      cursor: 'pointer',
                      border: isSelected ? '2px solid #ffffff' : undefined,
                      transform: isSelected ? 'scale(1.05)' : 'none',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '4px'
                    }}
                  >
                    <AlertTriangle style={{ width: '12px', height: '12px' }} />
                    Event #{ev.event_id} ({ev.start_time.toFixed(1)}s)
                  </button>
                );
              })
            )}
          </div>
        </div>
      </div>

      {/* Side Workspace Panel: Tabbed Risk Events & AI Recommendations */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {/* Navigation Tabs */}
        <div style={{ display: 'flex', borderBottom: '1px solid rgba(255,255,255,0.1)', paddingBottom: '8px', gap: '12px' }}>
          <button
            onClick={() => setActiveTab('RISKS')}
            style={{
              background: 'none',
              border: 'none',
              color: activeTab === 'RISKS' ? '#6366f1' : '#94a3b8',
              fontWeight: activeTab === 'RISKS' ? 700 : 500,
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              borderBottom: activeTab === 'RISKS' ? '2px solid #6366f1' : 'none',
              paddingBottom: '4px'
            }}
          >
            <AlertTriangle style={{ width: '16px', height: '16px' }} />
            Risk Events
          </button>

          <button
            onClick={() => setActiveTab('RECOMMENDATIONS')}
            style={{
              background: 'none',
              border: 'none',
              color: activeTab === 'RECOMMENDATIONS' ? '#10b981' : '#94a3b8',
              fontWeight: activeTab === 'RECOMMENDATIONS' ? 700 : 500,
              fontSize: '0.9rem',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              borderBottom: activeTab === 'RECOMMENDATIONS' ? '2px solid #10b981' : 'none',
              paddingBottom: '4px'
            }}
          >
            <Lightbulb style={{ width: '16px', height: '16px' }} />
            AI Recommendations ({recommendations.length})
          </button>
        </div>

        {/* Tab 1: Risk Event Detail */}
        {activeTab === 'RISKS' && (
          <div>
            {selectedEvent ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Risk Level</span>
                  <span className={selectedEvent.risk_level === 'HIGH' ? 'badge-high' : selectedEvent.risk_level === 'MEDIUM' ? 'badge-medium' : 'badge-low'}>
                    {selectedEvent.risk_level}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Risk Score</span>
                  <span style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc' }}>{selectedEvent.risk_score.toFixed(3)}</span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Time to Collision (TTC)</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600, color: selectedEvent.ttc && selectedEvent.ttc < 3.5 ? '#f43f5e' : '#10b981' }}>
                    {selectedEvent.ttc ? `${selectedEvent.ttc.toFixed(2)}s` : 'N/A (>30s)'}
                  </span>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Vehicles Involved</span>
                  <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#6366f1' }}>
                    {selectedEvent.vehicles.map(v => `V#${v}`).join(' & ')}
                  </span>
                </div>

                <div style={{ borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: '10px' }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8', marginBottom: '6px' }}>RISK REASONS</div>
                  <ul style={{ paddingLeft: '18px', fontSize: '0.8rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                    {selectedEvent.risk_reasons.map((r, i) => (
                      <li key={i}>{r}</li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px 10px', color: '#64748b', fontSize: '0.85rem' }}>
                <CheckCircle2 style={{ width: '32px', height: '32px', margin: '0 auto 8px', color: '#10b981', opacity: 0.8 }} />
                Click any timeline marker to inspect vehicle conflict metrics.
              </div>
            )}
          </div>
        )}

        {/* Tab 2: AI Road Safety Recommendations */}
        {activeTab === 'RECOMMENDATIONS' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', maxHeight: '480px', overflowY: 'auto', paddingRight: '4px' }}>
            {recommendations.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 10px', color: '#64748b', fontSize: '0.85rem' }}>
                No recommendations generated for this video.
              </div>
            ) : (
              recommendations.map((rec) => {
                const prioBadge = rec.priority === 'CRITICAL' || rec.priority === 'HIGH' ? 'badge-high' : rec.priority === 'MEDIUM' ? 'badge-medium' : 'badge-low';
                return (
                  <div key={rec.id} className="glass-card" style={{ padding: '14px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span className={prioBadge}>{rec.priority} PRIORITY</span>
                      <span style={{ fontSize: '0.7rem', fontWeight: 600, color: '#818cf8', background: 'rgba(99, 102, 241, 0.15)', padding: '2px 6px', borderRadius: '4px' }}>
                        {rec.source}
                      </span>
                    </div>

                    <h4 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc' }}>{rec.title}</h4>
                    <p style={{ fontSize: '0.8rem', color: '#cbd5e1' }}>{rec.summary}</p>

                    {/* Traceable Evidence */}
                    <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '8px 10px', borderRadius: '6px', border: '1px solid rgba(255,255,255,0.05)' }}>
                      <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#94a3b8', display: 'block', marginBottom: '4px' }}>OBSERVED EVIDENCE:</span>
                      <ul style={{ paddingLeft: '14px', fontSize: '0.75rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '2px' }}>
                        {rec.evidence.map((evItem, i) => (
                          <li key={i}>{evItem}</li>
                        ))}
                      </ul>
                    </div>

                    <div style={{ fontSize: '0.8rem', color: '#10b981', fontWeight: 500 }}>
                      <strong>Suggested Action:</strong> {rec.recommended_action}
                    </div>

                    {rec.limitations && (
                      <div style={{ fontSize: '0.7rem', color: '#64748b', fontStyle: 'italic' }}>
                        Limitations: {rec.limitations}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        )}
      </div>
    </div>
  );
};
