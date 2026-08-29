import React from 'react';
import { NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { LayoutDashboard, UploadCloud, Cpu, AlertTriangle, FileText, Settings, Users } from 'lucide-react';

export const Sidebar: React.FC = () => {
  const { user } = useAuth();

  const links = [
    { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard, roles: ['TRAFFIC_OPERATOR', 'ADMINISTRATOR', 'GOVERNMENT_AUTHORITY'] },
    { to: '/upload', label: 'Upload Video', icon: UploadCloud, roles: ['TRAFFIC_OPERATOR', 'ADMINISTRATOR'] },
    { to: '/jobs', label: 'Processing Jobs', icon: Cpu, roles: ['TRAFFIC_OPERATOR', 'ADMINISTRATOR'] },
    { to: '/risk-events', label: 'Risk Events', icon: AlertTriangle, roles: ['TRAFFIC_OPERATOR', 'ADMINISTRATOR', 'GOVERNMENT_AUTHORITY'] },
    { to: '/reports', label: 'Reports & Export', icon: FileText, roles: ['TRAFFIC_OPERATOR', 'ADMINISTRATOR', 'GOVERNMENT_AUTHORITY'] },
    { to: '/settings', label: 'Settings', icon: Settings, roles: ['TRAFFIC_OPERATOR', 'ADMINISTRATOR', 'GOVERNMENT_AUTHORITY'] },
    { to: '/admin/users', label: 'User Admin', icon: Users, roles: ['ADMINISTRATOR'] },
  ];

  return (
    <aside style={{ width: '240px', background: 'rgba(15, 23, 42, 0.95)', borderRight: '1px solid rgba(255, 255, 255, 0.08)', padding: '20px 12px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      {links.map((link) => {
        if (user && !link.roles.includes(user.role)) return null;
        const Icon = link.icon;
        return (
          <NavLink
            key={link.to}
            to={link.to}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              padding: '10px 16px',
              borderRadius: '8px',
              color: isActive ? '#ffffff' : '#94a3b8',
              background: isActive ? 'linear-gradient(90deg, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0.05) 100%)' : 'transparent',
              borderLeft: isActive ? '3px solid #6366f1' : '3px solid transparent',
              textDecoration: 'none',
              fontSize: '0.9rem',
              fontWeight: isActive ? 600 : 500,
              transition: 'all 0.15s ease'
            })}
          >
            <Icon style={{ width: '18px', height: '18px' }} />
            <span>{link.label}</span>
          </NavLink>
        );
      })}
    </aside>
  );
};
