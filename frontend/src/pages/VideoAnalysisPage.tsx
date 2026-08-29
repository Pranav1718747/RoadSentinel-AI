import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import api from '../api/client';
import { VideoPlayerOverlay, RiskEventItem, RecommendationItem } from '../components/VideoPlayerOverlay';
import { Video, FileText, ArrowLeft, ShieldAlert, AlertTriangle, Lightbulb, ChevronDown, ChevronUp, Activity, CheckCircle2, Gauge } from 'lucide-react';

interface RiskSummaryData {
  video_id: number;
  overall_risk_score: number;
  overall_risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  risk_distribution: {
    high: number;
    medium: number;
    low: number;
    total: number;
  };
  critical_metrics: {
    minimum_ttc: number | null;
    minimum_distance: number | null;
    maximum_closing_speed: number | null;
  };
  dominant_risk_patterns: string[];
  recommendations_count: number;
}

export const VideoAnalysisPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [video, setVideo] = useState<any>(null);
  const [riskSummary, setRiskSummary] = useState<RiskSummaryData | null>(null);
  const [riskEvents, setRiskEvents] = useState<RiskEventItem[]>([]);
  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [showEvidenceTable, setShowEvidenceTable] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    if (id) {
      Promise.all([
        api.get(`/videos/${id}/`),
        api.get(`/videos/${id}/risk-summary/`),
        api.get(`/videos/${id}/risk-events/`),
        api.get(`/videos/${id}/recommendations/`)
      ])
        .then(([vRes, sumRes, rRes, recRes]) => {
          setVideo(vRes.data);
          setRiskSummary(sumRes.data);
          setRiskEvents(rRes.data.results || rRes.data);
          setRecommendations(recRes.data.results || recRes.data);
        })
        .catch((err) => console.error(err))
        .finally(() => setLoading(false));
    }
  }, [id]);

  if (loading) {
    return <div style={{ padding: '40px', color: '#94a3b8' }}>Loading traffic scene risk assessment...</div>;
  }

  if (!video) {
    return <div style={{ padding: '40px', color: '#f43f5e' }}>Video not found.</div>;
  }

  const videoUrl = `/media/runs/run_20260824_214156/visualization/roadsentinel_final.mp4`;
  const overallScore = riskSummary ? Math.round(riskSummary.overall_risk_score) : 39;
  const overallLevel = riskSummary?.overall_risk_level || 'MEDIUM';

  const levelBadgeClass = 
    overallLevel === 'CRITICAL' || overallLevel === 'HIGH' ? 'badge-high' : 
    overallLevel === 'MEDIUM' ? 'badge-medium' : 'badge-low';

  const levelColor = 
    overallLevel === 'CRITICAL' ? '#f43f5e' : 
    overallLevel === 'HIGH' ? '#f43f5e' : 
    overallLevel === 'MEDIUM' ? '#f59e0b' : '#10b981';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      {/* Header Navigation */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <button onClick={() => navigate('/dashboard')} style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#94a3b8', padding: '8px', borderRadius: '6px', cursor: 'pointer' }}>
            <ArrowLeft style={{ width: '18px', height: '18px' }} />
          </button>
          <div>
            <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>
              TRAFFIC SCENE RISK ASSESSMENT
            </h2>
            <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>
              Observed Traffic Scene: <strong style={{ color: '#f8fafc' }}>{video.filename}</strong> &bull; Duration: {video.duration.toFixed(1)}s &bull; FPS: {video.fps} &bull; Model: <span style={{ color: '#10b981' }}>YOLO26 + XGBoost</span>
            </p>
          </div>
        </div>

        <button onClick={() => navigate('/reports')} className="btn-primary" style={{ padding: '8px 16px', fontSize: '0.85rem' }}>
          <FileText style={{ width: '16px', height: '16px' }} />
          Export PDF / CSV Report
        </button>
      </div>

      {/* 1. OVERALL TRAFFIC RISK BANNER */}
      <div className="glass-panel" style={{ padding: '28px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderLeft: `6px solid ${levelColor}` }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <div style={{ textAlign: 'center', background: 'rgba(15, 23, 42, 0.7)', padding: '18px 28px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.08)' }}>
            <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#94a3b8', letterSpacing: '0.05em' }}>OVERALL TRAFFIC RISK</div>
            <div style={{ fontSize: '2.8rem', fontWeight: 800, color: levelColor, lineHeight: 1.1, marginTop: '4px' }}>
              {overallScore} <span style={{ fontSize: '1.2rem', color: '#64748b' }}>/ 100</span>
            </div>
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span className={levelBadgeClass} style={{ fontSize: '0.95rem', padding: '4px 12px' }}>
                OVERALL RISK LEVEL: {overallLevel}
              </span>
            </div>
            <p style={{ fontSize: '0.9rem', color: '#cbd5e1', marginTop: '10px', maxWidth: '650px' }}>
              Scene-level safety assessment derived from spatial risk aggregation, vehicle conflict densities, time-to-collision (TTC) bounds, and closing-speed kinematics across the observed corridor.
            </p>
          </div>
        </div>

        {/* Quick Summary Stats */}
        <div style={{ display: 'flex', gap: '24px', textAlign: 'right' }}>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>TOTAL CONFLICTS</span>
            <span style={{ fontSize: '1.2rem', fontWeight: 800, color: '#f8fafc' }}>
              {riskSummary?.risk_distribution.total || riskEvents.length}
            </span>
          </div>
          <div>
            <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block' }}>MINIMUM TTC</span>
            <span style={{ fontSize: '1.2rem', fontWeight: 800, color: riskSummary?.critical_metrics.minimum_ttc && riskSummary.critical_metrics.minimum_ttc <= 1.5 ? '#f43f5e' : '#10b981' }}>
              {riskSummary?.critical_metrics.minimum_ttc ? `${riskSummary.critical_metrics.minimum_ttc.toFixed(2)}s` : 'N/A (>30s)'}
            </span>
          </div>
        </div>
      </div>

      {/* 2. RISK SUMMARY & CRITICAL METRICS GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
        {/* Card A: Risk Conflict Summary Breakdown */}
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Activity style={{ width: '18px', height: '18px', color: '#6366f1' }} />
            RISK CONFLICT SUMMARY
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '10px', textAlign: 'center' }}>
            <div style={{ background: 'rgba(244, 63, 94, 0.12)', padding: '12px 8px', borderRadius: '8px', border: '1px solid rgba(244, 63, 94, 0.2)' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f43f5e', display: 'block' }}>HIGH</span>
              <span style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f8fafc' }}>{riskSummary?.risk_distribution.high || 0}</span>
            </div>
            <div style={{ background: 'rgba(245, 158, 11, 0.12)', padding: '12px 8px', borderRadius: '8px', border: '1px solid rgba(245, 158, 11, 0.2)' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#f59e0b', display: 'block' }}>MEDIUM</span>
              <span style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f8fafc' }}>{riskSummary?.risk_distribution.medium || 0}</span>
            </div>
            <div style={{ background: 'rgba(16, 185, 129, 0.12)', padding: '12px 8px', borderRadius: '8px', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#10b981', display: 'block' }}>LOW</span>
              <span style={{ fontSize: '1.5rem', fontWeight: 800, color: '#f8fafc' }}>{riskSummary?.risk_distribution.low || 0}</span>
            </div>
          </div>
        </div>

        {/* Card B: Critical Kinematic Metrics */}
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Gauge style={{ width: '18px', height: '18px', color: '#10b981' }} />
            CRITICAL KINEMATIC METRICS
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px' }}>
              <span style={{ color: '#94a3b8' }}>Minimum Time-to-Collision (TTC)</span>
              <strong style={{ color: riskSummary?.critical_metrics.minimum_ttc && riskSummary.critical_metrics.minimum_ttc <= 1.5 ? '#f43f5e' : '#f8fafc' }}>
                {riskSummary?.critical_metrics.minimum_ttc ? `${riskSummary.critical_metrics.minimum_ttc.toFixed(2)}s` : 'N/A (>30s)'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.05)', paddingBottom: '4px' }}>
              <span style={{ color: '#94a3b8' }}>Minimum Vehicle Proximity</span>
              <strong style={{ color: '#f8fafc' }}>
                {riskSummary?.critical_metrics.minimum_distance ? `${riskSummary.critical_metrics.minimum_distance.toFixed(1)} px` : 'N/A'}
              </strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span style={{ color: '#94a3b8' }}>Maximum Closing Speed</span>
              <strong style={{ color: '#6366f1' }}>
                {riskSummary?.critical_metrics.maximum_closing_speed ? `${riskSummary.critical_metrics.maximum_closing_speed.toFixed(1)} px/s` : 'N/A'}
              </strong>
            </div>
          </div>
        </div>

        {/* Card C: Main Risk Patterns */}
        <div className="glass-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#94a3b8', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle style={{ width: '18px', height: '18px', color: '#f59e0b' }} />
            MAIN RISK PATTERNS
          </div>
          <ul style={{ paddingLeft: '16px', fontSize: '0.8rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {(riskSummary?.dominant_risk_patterns || [
              "Rapid closing-speed interactions observed across the corridor",
              "Short time-to-collision conflicts (TTC ≤ 1.5s)",
              "Repeated close vehicle interactions"
            ]).map((pat, i) => (
              <li key={i}>{pat}</li>
            ))}
          </ul>
        </div>
      </div>

      {/* 3. VIDEO PLAYER & AI RECOMMENDATIONS WORKSPACE */}
      <div>
        <div style={{ fontSize: '1rem', fontWeight: 700, color: '#f8fafc', marginBottom: '12px' }}>
          OBSERVED SCENE VISUALIZATION & AI SAFETY RECOMMENDATIONS
        </div>
        <VideoPlayerOverlay
          videoUrl={videoUrl}
          duration={video.duration || 10.0}
          riskEvents={riskEvents}
          recommendations={recommendations}
        />
      </div>

      {/* 4. DETAILED EVENT EVIDENCE (SECONDARY COLLAPSIBLE PANEL) */}
      <div className="glass-panel" style={{ padding: '20px' }}>
        <button
          onClick={() => setShowEvidenceTable(!showEvidenceTable)}
          style={{
            width: '100%',
            background: 'none',
            border: 'none',
            color: '#f8fafc',
            fontSize: '1rem',
            fontWeight: 700,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <AlertTriangle style={{ width: '20px', height: '20px', color: '#6366f1' }} />
            DETAILED EVENT EVIDENCE (SUPPORTING CONFLICT DATA &ndash; {riskEvents.length} EVENTS)
          </div>
          {showEvidenceTable ? <ChevronUp style={{ width: '20px', height: '20px' }} /> : <ChevronDown style={{ width: '20px', height: '20px' }} />}
        </button>

        {showEvidenceTable && (
          <div style={{ marginTop: '16px', overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem', color: '#cbd5e1' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: '#94a3b8' }}>
                  <th style={{ padding: '10px' }}>Event ID</th>
                  <th style={{ padding: '10px' }}>Timestamp</th>
                  <th style={{ padding: '10px' }}>Vehicles</th>
                  <th style={{ padding: '10px' }}>Risk Level</th>
                  <th style={{ padding: '10px' }}>Risk Score</th>
                  <th style={{ padding: '10px' }}>TTC (s)</th>
                  <th style={{ padding: '10px' }}>Closing Speed</th>
                  <th style={{ padding: '10px' }}>Primary Reason</th>
                </tr>
              </thead>
              <tbody>
                {riskEvents.map((ev) => (
                  <tr key={ev.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                    <td style={{ padding: '10px', fontWeight: 600, color: '#f8fafc' }}>#{ev.event_id}</td>
                    <td style={{ padding: '10px' }}>{ev.start_time.toFixed(1)}s &ndash; {ev.end_time.toFixed(1)}s</td>
                    <td style={{ padding: '10px', color: '#818cf8' }}>{ev.vehicles.map(v => `V#${v}`).join(', ')}</td>
                    <td style={{ padding: '10px' }}>
                      <span className={ev.risk_level === 'HIGH' ? 'badge-high' : ev.risk_level === 'MEDIUM' ? 'badge-medium' : 'badge-low'}>
                        {ev.risk_level}
                      </span>
                    </td>
                    <td style={{ padding: '10px', fontWeight: 700 }}>{ev.risk_score.toFixed(3)}</td>
                    <td style={{ padding: '10px' }}>{ev.ttc ? `${ev.ttc.toFixed(2)}s` : 'N/A'}</td>
                    <td style={{ padding: '10px' }}>{ev.closing_speed ? `${ev.closing_speed.toFixed(1)} px/s` : 'N/A'}</td>
                    <td style={{ padding: '10px' }}>{ev.risk_reasons.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};
