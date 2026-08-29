import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import { Cpu, CheckCircle2, Clock, AlertCircle, PlayCircle, RefreshCw, XCircle, Zap, Eye, Gauge } from 'lucide-react';

interface ProcessingJob {
  id: number;
  video: number;
  video_filename: string;
  status: 'PENDING' | 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
  progress: number;
  current_stage: string;
  stage_progress: number;
  overall_progress: number;
  current_frame: number;
  total_frames: number;
  video_position_seconds: number;
  video_duration_seconds: number;
  elapsed_seconds: number;
  processing_fps: number;
  eta_seconds: number | null;
  processed_detections: number;
  processed_tracks: number;
  processed_trajectories: number;
  processed_interactions: number;
  last_progress_update: string | null;
  started_at: string;
  completed_at: string | null;
  error_message: string | null;
}

export const JobsPage: React.FC = () => {
  const [jobs, setJobs] = useState<ProcessingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [nowTs, setNowTs] = useState(Date.now());
  const navigate = useNavigate();

  const fetchJobs = () => {
    api.get('/jobs/')
      .then((res) => setJobs(res.data.results || res.data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(() => {
      fetchJobs();
      setNowTs(Date.now());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleCancel = async (jobId: number) => {
    try {
      await api.post(`/jobs/${jobId}/cancel/`);
      fetchJobs();
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>Real-Time Processing Monitor</h2>
          <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>Live frame-level progress, speed (FPS), and ETA tracker</p>
        </div>
        <button onClick={fetchJobs} className="glass-card" style={{ padding: '8px 14px', color: '#94a3b8', border: '1px solid rgba(255,255,255,0.1)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <RefreshCw style={{ width: '16px', height: '16px' }} />
          Refresh Status
        </button>
      </div>

      <div className="glass-panel" style={{ padding: '20px' }}>
        {loading && jobs.length === 0 ? (
          <div style={{ padding: '20px', color: '#94a3b8' }}>Loading processing jobs...</div>
        ) : jobs.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px 10px', color: '#64748b' }}>
            No processing jobs found. Upload a traffic video to start pipeline analysis.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            {jobs.map((job) => {
              const isCompleted = job.status === 'COMPLETED';
              const isFailed = job.status === 'FAILED';
              const isCancelled = job.status === 'CANCELLED';

              const statusBadge = isCompleted ? 'badge-low' : isFailed || isCancelled ? 'badge-high' : 'badge-medium';
              const lastUpdatedSec = job.last_progress_update ? Math.max(0, ((nowTs - new Date(job.last_progress_update).getTime()) / 1000)).toFixed(1) : 'N/A';
              
              const currentStageDisplay = isCompleted ? 'COMPLETED' : job.current_stage.replace('_', ' ');
              const stageProgDisplay = isCompleted ? '100.0' : (job.stage_progress ? job.stage_progress.toFixed(1) : (job.progress || 0).toFixed(1));
              const overallProgDisplay = isCompleted ? '100.0' : (job.overall_progress ? job.overall_progress.toFixed(1) : (job.progress || 0).toFixed(1));

              // Frame & Video position displays (handle historical N/A vs real numbers)
              const frameDisplay = job.total_frames && job.total_frames > 0 
                ? `${job.current_frame || job.total_frames} / ${job.total_frames}` 
                : 'N/A';
                
              const videoTimeDisplay = job.video_duration_seconds && job.video_duration_seconds > 0 
                ? `${(job.video_position_seconds || job.video_duration_seconds).toFixed(2)}s / ${job.video_duration_seconds.toFixed(2)}s` 
                : 'N/A';

              const speedDisplay = job.processing_fps && job.processing_fps > 0 
                ? `${job.processing_fps.toFixed(1)} FPS` 
                : 'N/A';

              const etaDisplay = isCompleted 
                ? 'Completed' 
                : (job.eta_seconds !== null && job.eta_seconds !== undefined ? `${job.eta_seconds.toFixed(1)}s` : 'Calculating...');

              return (
                <div key={job.id} className="glass-card" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
                  {/* Job Header */}
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <div style={{ width: '38px', height: '38px', borderRadius: '8px', background: isCompleted ? 'rgba(16, 185, 129, 0.15)' : 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: isCompleted ? '#10b981' : '#6366f1' }}>
                        {isCompleted ? <CheckCircle2 style={{ width: '22px', height: '22px' }} /> : <Cpu style={{ width: '22px', height: '22px' }} />}
                      </div>
                      <div>
                        <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: '#f8fafc' }}>
                          Job #{job.id} — {job.video_filename}
                        </h3>
                        <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>
                          Stage: <strong style={{ color: isCompleted ? '#10b981' : '#818cf8' }}>{currentStageDisplay}</strong> &bull; {isCompleted ? 'Finished' : `Heartbeat: ${lastUpdatedSec}s ago`}
                        </div>
                      </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                      <span className={statusBadge}>{job.status}</span>
                      {job.status === 'PROCESSING' && (
                        <button
                          onClick={() => handleCancel(job.id)}
                          style={{ background: 'rgba(244, 63, 94, 0.15)', border: '1px solid rgba(244, 63, 94, 0.3)', color: '#f43f5e', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '4px' }}
                        >
                          <XCircle style={{ width: '14px', height: '14px' }} />
                          Cancel Job
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Progress Bar Dual Indicators */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px' }}>
                      <span>Stage Progress ({currentStageDisplay}): <strong style={{ color: '#f8fafc' }}>{stageProgDisplay}%</strong></span>
                      <span>Overall Pipeline Progress: <strong style={{ color: isCompleted ? '#10b981' : '#6366f1' }}>{overallProgDisplay}%</strong></span>
                    </div>
                    <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.08)', borderRadius: '4px', overflow: 'hidden' }}>
                      <div style={{ width: `${overallProgDisplay}%`, height: '100%', background: isFailed ? '#f43f5e' : 'linear-gradient(90deg, #6366f1 0%, #10b981 100%)', transition: 'width 0.3s ease' }} />
                    </div>
                  </div>

                  {/* Real-time Grid Metrics (Frame, Video Position, Speed, ETA, Detections) */}
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', background: 'rgba(15, 23, 42, 0.5)', padding: '14px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                    <div>
                      <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block', marginBottom: '2px' }}>FRAME COUNTER</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc' }}>
                        {frameDisplay}
                      </span>
                    </div>

                    <div>
                      <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block', marginBottom: '2px' }}>VIDEO TIME</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#f8fafc' }}>
                        {videoTimeDisplay}
                      </span>
                    </div>

                    <div>
                      <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block', marginBottom: '2px' }}>ELAPSED TIME</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#cbd5e1' }}>
                        {job.elapsed_seconds && job.elapsed_seconds > 0 ? `${job.elapsed_seconds.toFixed(1)}s` : 'N/A'}
                      </span>
                    </div>

                    <div>
                      <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block', marginBottom: '2px' }}>AVERAGE SPEED</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#10b981' }}>
                        {speedDisplay}
                      </span>
                    </div>

                    <div>
                      <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block', marginBottom: '2px' }}>ESTIMATED ETA</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: 700, color: isCompleted ? '#10b981' : '#f59e0b' }}>
                        {etaDisplay}
                      </span>
                    </div>

                    <div>
                      <span style={{ fontSize: '0.7rem', color: '#64748b', display: 'block', marginBottom: '2px' }}>DETECTIONS</span>
                      <span style={{ fontSize: '0.95rem', fontWeight: 700, color: '#6366f1' }}>
                        {job.processed_detections ? job.processed_detections.toLocaleString() : 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Error Message Box */}
                  {isFailed && job.error_message && (
                    <div style={{ background: 'rgba(244, 63, 94, 0.15)', border: '1px solid rgba(244, 63, 94, 0.3)', color: '#f43f5e', padding: '10px 14px', borderRadius: '6px', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <AlertCircle style={{ width: '16px', height: '16px', flexShrink: 0 }} />
                      <span>Error: {job.error_message}</span>
                    </div>
                  )}

                  {/* Completed Job Workspace Action Button */}
                  {isCompleted && (
                    <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                      <button
                        onClick={() => navigate(`/videos/${job.video}`)}
                        className="btn-primary"
                        style={{ padding: '6px 14px', fontSize: '0.8rem' }}
                      >
                        <PlayCircle style={{ width: '16px', height: '16px' }} />
                        Open Video Workspace
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
