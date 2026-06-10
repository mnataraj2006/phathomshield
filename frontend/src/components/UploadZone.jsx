import React, { useCallback, useRef, useState } from 'react';

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

export default function UploadZone({ id, onFileSelect, accept = '.dcm', label = 'DICOM File' }) {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [dropping, setDropping] = useState(false);

  const handleFile = useCallback((f) => {
    if (!f) return;
    // Trigger drop visual effect
    setDropping(true);
    setTimeout(() => setDropping(false), 600);
    setFile(f);
    onFileSelect(f);
  }, [onFileSelect]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleRemove = useCallback((e) => {
    e.stopPropagation();
    setFile(null);
    if (inputRef.current) inputRef.current.value = '';
    onFileSelect(null);
  }, [onFileSelect]);

  return (
    <div>
      <div
        id={id}
        className={`upload-zone${dragging ? ' dragover' : ''}${file ? ' has-file' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        role="button"
        tabIndex={0}
        aria-label={`Upload ${label}`}
        onClick={() => !file && inputRef.current?.click()}
        onKeyDown={e => e.key === 'Enter' && !file && inputRef.current?.click()}
        style={{
          transition: 'all 0.3s ease',
          ...(dropping ? {
            boxShadow: '0 0 60px rgba(0, 245, 255, 0.4), inset 0 0 40px rgba(0, 245, 255, 0.15)',
            transform: 'scale(1.01)',
          } : {})
        }}
      >
        {!file && (
          <input
            ref={inputRef}
            type="file"
            accept={accept}
            aria-hidden="true"
            onChange={e => handleFile(e.target.files[0])}
          />
        )}

        {file ? (
          <>
            <span className="upload-icon" style={{ filter: 'drop-shadow(0 0 20px rgba(16,185,129,0.8))' }}>
              ✅
            </span>
            <p className="upload-title" style={{ color: 'var(--green-bright)' }}>File Ready for Analysis</p>
            <p className="upload-hint">
              Click <span>✕</span> below to remove and choose another file
            </p>
          </>
        ) : (
          <>
            <span
              className="upload-icon"
              style={{
                filter: dragging
                  ? 'drop-shadow(0 0 30px rgba(0,245,255,1))'
                  : 'drop-shadow(0 0 12px rgba(0,245,255,0.4))'
              }}
            >
              {dragging ? '📂' : '📁'}
            </span>
            <p className="upload-title">
              {dragging ? '✦ Drop it to begin analysis ✦' : 'Drag & Drop your DICOM file'}
            </p>
            <p className="upload-hint">
              or click to browse — accepts <span>{accept}</span> files
            </p>

            {/* Decorative corner accents */}
            {['top-left','top-right','bottom-left','bottom-right'].map(pos => (
              <div key={pos} style={{
                position: 'absolute',
                [pos.includes('top') ? 'top' : 'bottom']: '12px',
                [pos.includes('left') ? 'left' : 'right']: '12px',
                width: '20px',
                height: '20px',
                borderTop: pos.includes('top') ? '2px solid rgba(0,245,255,0.3)' : 'none',
                borderBottom: pos.includes('bottom') ? '2px solid rgba(0,245,255,0.3)' : 'none',
                borderLeft: pos.includes('left') ? '2px solid rgba(0,245,255,0.3)' : 'none',
                borderRight: pos.includes('right') ? '2px solid rgba(0,245,255,0.3)' : 'none',
                opacity: dragging ? 1 : 0.4,
                transition: 'opacity 0.3s',
                pointerEvents: 'none',
              }} />
            ))}
          </>
        )}
      </div>

      {file && (
        <div className="file-selected">
          <span className="file-icon">🗂️</span>
          <div className="file-info">
            <p className="file-name">{file.name}</p>
            <p className="file-size">{formatBytes(file.size)} · DICOM Medical Image</p>
          </div>
          <button
            className="file-remove"
            aria-label="Remove selected file"
            onClick={handleRemove}
            id={`${id}-remove`}
            title="Remove file"
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
