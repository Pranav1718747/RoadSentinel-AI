import React from 'react';
import { Users, ShieldAlert, CheckCircle2 } from 'lucide-react';

export const AdminUsersPage: React.FC = () => {
  const users = [
    { id: 1, username: 'operator', email: 'operator@roadsentinel.ai', role: 'TRAFFIC_OPERATOR', org: 'City Traffic Control', status: 'Active' },
    { id: 2, username: 'admin', email: 'admin@roadsentinel.ai', role: 'ADMINISTRATOR', org: 'RoadSentinel HQ', status: 'Active' },
    { id: 3, username: 'authority', email: 'authority@roadsentinel.ai', role: 'GOVERNMENT_AUTHORITY', org: 'Dept of Transportation', status: 'Active' },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>System User Management</h2>
        <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>Administrator panel for role assignment and monitoring</p>
      </div>

      <div className="glass-panel" style={{ padding: '20px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left', color: '#94a3b8' }}>
              <th style={{ padding: '12px' }}>User ID</th>
              <th style={{ padding: '12px' }}>Username</th>
              <th style={{ padding: '12px' }}>Email</th>
              <th style={{ padding: '12px' }}>Role</th>
              <th style={{ padding: '12px' }}>Organization</th>
              <th style={{ padding: '12px' }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                <td style={{ padding: '12px', fontWeight: 600 }}>#{u.id}</td>
                <td style={{ padding: '12px', color: '#f8fafc', fontWeight: 600 }}>{u.username}</td>
                <td style={{ padding: '12px', color: '#94a3b8' }}>{u.email}</td>
                <td style={{ padding: '12px' }}>
                  <span className={u.role === 'ADMINISTRATOR' ? 'badge-high' : u.role === 'GOVERNMENT_AUTHORITY' ? 'badge-medium' : 'badge-low'}>
                    {u.role}
                  </span>
                </td>
                <td style={{ padding: '12px', color: '#cbd5e1' }}>{u.org}</td>
                <td style={{ padding: '12px', color: '#10b981', fontWeight: 600 }}>{u.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
