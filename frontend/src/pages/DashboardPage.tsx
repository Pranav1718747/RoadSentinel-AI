import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import { Video, AlertTriangle, ShieldCheck, Cpu, Clock, ChevronRight } from 'lucide-react';

interface SummaryData {
  total_videos: number;
  completed_jobs: number;
  processing_jobs: number;
  total_risk_events: number;
  risk_counts: {
    HIGH: number;
    MEDIUM: number;
    LOW: number;
  };
  averages: {
    avg_ttc: number;
    avg_minimum_distance: number;
    avg_risk_score: number;
  };
}

export const DashboardPage: React.FC = () => {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    api.get('/dashboard/summary/')
      .then((res) => setSummary(res.data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div style={{ padding: '40px', color: '#94a3b8' }}>Loading dashboard analytics...</div>;
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
      <div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>System Overview Dashboard</h2>
        <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>Real-time traffic risk analysis & pipeline metrics</p>
      </div>

      {/* Top Metrics Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8' }}>TOTAL VIDEOS</span>
            <Video style={{ width: '20px', height: '20px', color: '#6366f1' }} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc' }}>{summary?.total_videos || 0}</div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>Ingested traffic footage</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8' }}>HIGH RISK CONFLICTS</span>
            <AlertTriangle style={{ width: '20px', height: '20px', color: '#f43f5e' }} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f43f5e' }}>{summary?.risk_counts.HIGH || 0}</div>
          <div style={{ fontSize: '0.75rem', color: '#f43f5e', marginTop: '4px' }}>Critical safety events</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8' }}>MEDIUM RISK</span>
            <AlertTriangle style={{ width: '20px', height: '20px', color: '#f59e0b' }} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f59e0b' }}>{summary?.risk_counts.MEDIUM || 0}</div>
          <div style={{ fontSize: '0.75rem', color: '#f59e0b', marginTop: '4px' }}>Moderate conflicts</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8' }}>LOW RISK / SAFE</span>
            <ShieldCheck style={{ width: '20px', height: '20px', color: '#10b981' }} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#10b981' }}>{summary?.risk_counts.LOW || 0}</div>
          <div style={{ fontSize: '0.75rem', color: '#10b981', marginTop: '4px' }}>Normal traffic interactions</div>
        </div>

        <div className="glass-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#94a3b8' }}>PROCESSING JOBS</span>
            <Cpu style={{ width: '20px', height: '20px', color: '#818cf8' }} />
          </div>
          <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#f8fafc' }}>{summary?.processing_jobs || 0}</div>
          <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '4px' }}>Active Celery ML tasks</div>
        </div>
      </div>

      {/* Physics Averages & System Health Card */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: '20px' }}>
        <div className="glass-panel" style={{ padding: '24px' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '16px' }}>Physics Risk Indicator Averages</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px', textAlign: 'center' }}>
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>AVERAGE TTC</span>
              <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#6366f1' }}>{summary?.averages.avg_ttc}s</span>
            </div>
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>MIN DISTANCE</span>
              <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f8fafc' }}>{summary?.averages.avg_minimum_distance} px</span>
            </div>
            <div style={{ background: 'rgba(15, 23, 42, 0.5)', padding: '16px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
              <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block', marginBottom: '4px' }}>RISK SCORE AVG</span>
              <span style={{ fontSize: '1.4rem', fontWeight: 700, color: '#f59e0b' }}>{summary?.averages.avg_risk_score}</span>
            </div>
          </div>
        </div>

        <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700, marginBottom: '8px' }}>Pipeline Control</h3>
            <p style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Analyze new video footage with YOLO26 + XGBoost pipeline.</p>
          </div>
          <button onClick={() => navigate('/upload')} className="btn-primary" style={{ justifyContent: 'center', marginTop: '16px' }}>
            Upload Video & Analysis
            <ChevronRight style={{ width: '18px', height: '18px' }} />
          </button>
        </div>
      </div>
    </div>
  );
};
