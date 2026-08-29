import React from 'react';
import { useAuth } from '../context/AuthContext';
import { LogOut, Shield, User, Cpu } from 'lucide-react';

export const Navbar: React.FC = () => {
  const { user, logout } = useAuth();

  return (
    <header className="glass-panel" style={{ borderRadius: 0, borderBottom: '1px solid rgba(255,255,255,0.08)', padding: '12px 24px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', zIndex: 10 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
        <div style={{ width: '36px', height: '36px', borderRadius: '8px', background: 'linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Shield style={{ color: '#fff', width: '22px', height: '22px' }} />
        </div>
        <div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.02em', background: 'linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
            ROADSENTINEL AI
          </h1>
          <p style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 500 }}>
            Aerial Traffic Risk System &bull; <span style={{ color: '#10b981' }}>YOLO26 Active</span>
          </p>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
        {user && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc' }}>{user.username}</div>
              <span className={user.role === 'ADMINISTRATOR' ? 'badge-high' : user.role === 'GOVERNMENT_AUTHORITY' ? 'badge-medium' : 'badge-low'}>
                {user.role.replace('_', ' ')}
              </span>
            </div>
            <button 
              onClick={logout}
              style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: '#94a3b8', padding: '8px 12px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}
              title="Logout"
            >
              <LogOut style={{ width: '16px', height: '16px' }} />
              <span style={{ fontSize: '0.8rem' }}>Logout</span>
            </button>
          </div>
        )}
      </div>
    </header>
  );
};
