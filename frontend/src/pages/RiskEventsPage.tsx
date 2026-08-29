import React, { useState, useEffect } from 'react';
import api from '../api/client';
import { AlertTriangle, Search, Filter } from 'lucide-react';

export const RiskEventsPage: React.FC = () => {
  const [events, setEvents] = useState<any[]>([]);
  const [filterLevel, setFilterLevel] = useState<string>('ALL');
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.get('/dashboard/timeline/')
      .then((res) => setEvents(res.data.timeline_events || []))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, []);

  const filteredEvents = events.filter((ev) => {
    const matchesLevel = filterLevel === 'ALL' || ev.risk_level === filterLevel;
    const matchesSearch = searchTerm === '' || ev.event_id.toString().includes(searchTerm) || ev.risk_level.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>Risk Events Directory</h2>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>Search and filter detected vehicle conflict interactions</p>
        </div>

        <div style={{ display: 'flex', gap: '12px' }}>
          <div style={{ position: 'relative' }}>
            <Search style={{ position: 'absolute', left: '10px', top: '10px', width: '16px', height: '16px', color: '#64748b' }} />
            <input
              type="text"
              placeholder="Search event ID or level..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{ padding: '8px 12px 8px 34px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', color: '#fff', fontSize: '0.85rem' }}
            />
          </div>

          <select
            value={filterLevel}
            onChange={(e) => setFilterLevel(e.target.value)}
            style={{ padding: '8px 12px', background: 'rgba(15, 23, 42, 0.6)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '6px', color: '#fff', fontSize: '0.85rem' }}
          >
            <option value="ALL">All Risk Levels</option>
            <option value="HIGH">High Risk</option>
            <option value="MEDIUM">Medium Risk</option>
            <option value="LOW">Low Risk</option>
          </select>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '20px' }}>
        {loading ? (
          <div style={{ padding: '20px', color: '#94a3b8' }}>Loading risk events...</div>
        ) : filteredEvents.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>No risk events match the filter criteria.</div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.08)', textAlign: 'left', color: '#94a3b8' }}>
                <th style={{ padding: '12px' }}>Event ID</th>
                <th style={{ padding: '12px' }}>Time Window</th>
                <th style={{ padding: '12px' }}>Vehicles Involved</th>
                <th style={{ padding: '12px' }}>Risk Level</th>
                <th style={{ padding: '12px' }}>Risk Score</th>
              </tr>
            </thead>
            <tbody>
              {filteredEvents.map((ev) => (
                <tr key={ev.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                  <td style={{ padding: '12px', fontWeight: 600 }}>Event #{ev.event_id}</td>
                  <td style={{ padding: '12px', color: '#cbd5e1' }}>{ev.start_time.toFixed(1)}s – {ev.end_time.toFixed(1)}s</td>
                  <td style={{ padding: '12px', color: '#818cf8', fontWeight: 600 }}>
                    {ev.vehicles ? ev.vehicles.map((v: any) => `V#${v}`).join(' & ') : 'V#1 & V#2'}
                  </td>
                  <td style={{ padding: '12px' }}>
                    <span className={ev.risk_level === 'HIGH' ? 'badge-high' : ev.risk_level === 'MEDIUM' ? 'badge-medium' : 'badge-low'}>
                      {ev.risk_level}
                    </span>
                  </td>
                  <td style={{ padding: '12px', fontWeight: 700, color: '#f8fafc' }}>{ev.risk_score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
};
