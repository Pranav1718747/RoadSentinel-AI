import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../api/client';
import { UploadCloud, FileVideo, AlertCircle, CheckCircle2 } from 'lucide-react';

export const UploadPage: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const ext = file.name.split('.').pop()?.toLowerCase();
      if (!['mp4', 'avi', 'mov', 'mkv'].includes(ext || '')) {
        setError('Invalid file format. Please upload an MP4, AVI, or MOV video.');
        setSelectedFile(null);
        return;
      }
      setError('');
      setSelectedFile(file);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setError('');
    setProgress(10);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const res = await api.post('/videos/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (evt) => {
          if (evt.total) {
            const p = Math.round((evt.loaded * 90) / evt.total);
            setProgress(p);
          }
        }
      });
      setProgress(100);
      setTimeout(() => {
        navigate('/jobs');
      }, 800);
    } catch (err: any) {
      setError(err.response?.data?.message || 'Video upload failed. Please try again.');
      setIsUploading(false);
    }
  };

  return (
    <div style={{ maxWidth: '640px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '20px' }}>
      <div>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#f8fafc' }}>Upload Traffic Video</h2>
        <p style={{ fontSize: '0.85rem', color: '#94a3b8', marginTop: '2px' }}>
          Select 4K or HD traffic footage for YOLO26 detection & risk analysis
        </p>
      </div>

      {error && (
        <div style={{ background: 'rgba(244, 63, 94, 0.15)', border: '1px solid rgba(244, 63, 94, 0.3)', color: '#f43f5e', padding: '12px', borderRadius: '8px', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
          <AlertCircle style={{ width: '18px', height: '18px' }} />
          <span>{error}</span>
        </div>
      )}

      <div className="glass-panel" style={{ padding: '40px 24px', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', border: '2px dashed rgba(99, 102, 241, 0.4)' }}>
        <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'rgba(99, 102, 241, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '16px', color: '#6366f1' }}>
          <UploadCloud style={{ width: '32px', height: '32px' }} />
        </div>

        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f8fafc', marginBottom: '6px' }}>
          Drag & Drop Video File Here
        </h3>
        <p style={{ fontSize: '0.8rem', color: '#64748b', marginBottom: '20px' }}>
          Supports MP4, AVI, MOV up to 500MB (4K resolution supported)
        </p>

        <input
          type="file"
          accept="video/mp4,video/avi,video/quicktime,video/x-matroska"
          onChange={handleFileSelect}
          style={{ display: 'none' }}
          id="video-upload-input"
        />

        <label htmlFor="video-upload-input" className="btn-primary" style={{ cursor: 'pointer' }}>
          <FileVideo style={{ width: '18px', height: '18px' }} />
          Browse Video File
        </label>

        {selectedFile && (
          <div style={{ marginTop: '24px', padding: '12px 20px', background: 'rgba(15, 23, 42, 0.6)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', display: 'flex', alignItems: 'center', gap: '12px' }}>
            <FileVideo style={{ width: '20px', height: '20px', color: '#6366f1' }} />
            <div style={{ textAlign: 'left' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#f8fafc' }}>{selectedFile.name}</div>
              <div style={{ fontSize: '0.75rem', color: '#64748b' }}>{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</div>
            </div>
          </div>
        )}

        {isUploading && (
          <div style={{ width: '100%', marginTop: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '6px' }}>
              <span>Uploading & Queuing Celery Task...</span>
              <span>{progress}%</span>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ width: `${progress}%`, height: '100%', background: '#6366f1', transition: 'width 0.2s ease' }} />
            </div>
          </div>
        )}

        {selectedFile && !isUploading && (
          <button onClick={handleUpload} className="btn-primary" style={{ marginTop: '24px', width: '100%', justifyContent: 'center' }}>
            Start Pipeline Processing
          </button>
        )}
      </div>
    </div>
  );
};
