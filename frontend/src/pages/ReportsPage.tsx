import React, { useState, useEffect } from 'react';
import api from '../api/client';
import { FileText, Download, FileSpreadsheet, PlusCircle } from 'lucide-react';

export const ReportsPage: React.FC = () => {
  const [reports, setReports] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchReports = () => {
    api.get('/reports/')
      .then((res) => setReports(res.data.results || res.data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchReports();
  }, []);

  const handleGenerateReport = async (reportType: 'PDF' | 'CSV') => {
    try {
      await api.post('/videos/1/reports/', { report_type: reportType });
      fetchReports();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>Reports & Data Export</h2>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>Download executive PDF summaries and raw CSV risk logs</p>
        </div>

        <div style={{ display: 'flex', gap: '10px' }}>
          <button onClick={() => handleGenerateReport('PDF')} className="btn-primary" style={{ fontSize: '0.85rem' }}>
            <FileText style={{ width: '16px', height: '16px' }} />
            Generate PDF Summary
          </button>
          <button onClick={() => handleGenerateReport('CSV')} className="glass-card" style={{ padding: '8px 14px', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
            <FileSpreadsheet style={{ width: '16px', height: '16px' }} />
            Export CSV
          </button>
        </div>
      </div>

      <div className="glass-panel" style={{ padding: '20px' }}>
        {loading ? (
          <div style={{ padding: '20px', color: '#94a3b8' }}>Loading generated reports...</div>
        ) : reports.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
            No reports generated yet. Click "Generate PDF Summary" to create your first report.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {reports.map((rep) => (
              <div key={rep.id} className="glass-card" style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                  {rep.report_type === 'PDF' ? (
                    <FileText style={{ width: '24px', height: '24px', color: '#f43f5e' }} />
                  ) : (
                    <FileSpreadsheet style={{ width: '24px', height: '24px', color: '#10b981' }} />
                  )}
                  <div>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600, color: '#f8fafc' }}>
                      Report #{rep.id} ({rep.report_type}) — {rep.video_filename || 'traffic.mp4'}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                      Created at: {new Date(rep.created_at).toLocaleString()}
                    </div>
                  </div>
                </div>

                <a
                  href={`/api/reports/${rep.id}/download/`}
                  target="_blank"
                  rel="noreferrer"
                  className="btn-primary"
                  style={{ padding: '6px 12px', fontSize: '0.8rem', textDecoration: 'none' }}
                >
                  <Download style={{ width: '14px', height: '14px' }} />
                  Download
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
