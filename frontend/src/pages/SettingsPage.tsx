import React from 'react';
import { useAuth } from '../context/AuthContext';
import { Settings, User, Shield, Cpu } from 'lucide-react';

export const SettingsPage: React.FC = () => {
  const { user } = useAuth();

  return (
    <div style={{ maxWidth: '640px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>System Settings & Profile</h2>
        <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>User credentials, active role, and model parameters</p>
      </div>

      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>Active Account Profile</h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: '120px 1fr', gap: '12px', fontSize: '0.85rem' }}>
          <span style={{ color: '#94a3b8' }}>Username:</span>
          <span style={{ fontWeight: 600, color: '#f8fafc' }}>{user?.username}</span>

          <span style={{ color: '#94a3b8' }}>Email:</span>
          <span style={{ color: '#cbd5e1' }}>{user?.email || 'operator@roadsentinel.ai'}</span>

          <span style={{ color: '#94a3b8' }}>System Role:</span>
          <span className="badge-low" style={{ width: 'fit-content' }}>{user?.role}</span>

          <span style={{ color: '#94a3b8' }}>Organization:</span>
          <span style={{ color: '#cbd5e1' }}>{user?.organization || 'City Traffic Control'}</span>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>Frozen ML Pipeline Configuration</h3>
        <div style={{ fontSize: '0.85rem', color: '#94a3b8', display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div>Object Detector: <strong style={{ color: '#10b981' }}>YOLO26 Small (yolo26s.pt)</strong></div>
          <div>Multi-Object Tracker: <strong style={{ color: '#6366f1' }}>ByteTrack (bytetrack.yaml)</strong></div>
          <div>Inference Resolution: <strong style={{ color: '#f8fafc' }}>1280 px</strong></div>
          <div>Risk Model Classifier: <strong style={{ color: '#f59e0b' }}>XGBoost (risk_model.pkl)</strong></div>
        </div>
      </div>
    </div>
  );
};
