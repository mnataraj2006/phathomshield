import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import DetectWorkspace from './components/DetectWorkspace';
import RecoverWorkspace from './components/RecoverWorkspace';
import CaseArchive from './components/CaseArchive';
import { API_BASE } from './utils/api';

export default function App() {
  const [activeView, setActiveView] = useState('dashboard'); // 'dashboard' | 'detect' | 'recover' | 'archive'

  return (
    <div className="app-shell" id="app-shell">
      {/* Ambient Background */}
      <div className="ambient-bg" aria-hidden="true">
        <div className="ambient-orb orb-1" />
        <div className="ambient-orb orb-2" />
        <div className="ambient-orb orb-3" />
        <div className="ambient-grid" />
        <div className="ambient-scanline" />
      </div>

      {/* Sidebar Navigation */}
      <Sidebar activeView={activeView} onNavigate={setActiveView} />

      {/* Main Workspace */}
      <div className="workspace" id="main-workspace">
        {/* Top Bar */}
        <header className="topbar" role="banner">
          <div className="topbar-left">
            <div className="topbar-breadcrumb">
              <span className="breadcrumb-root">PhantomaShield</span>
              <span className="breadcrumb-sep">›</span>
              <span className="breadcrumb-current">
              {activeView === 'dashboard' ? 'Overview'
                : activeView === 'detect'  ? 'Forensic Detection'
                : activeView === 'recover' ? 'AI Recovery'
                : 'Case Archive'}
              </span>
            </div>
          </div>
          <div className="topbar-right">
            <div className="topbar-status" aria-label="System online">
              <span className="status-pulse" aria-hidden="true" />
              <span className="status-text">All Systems Operational</span>
            </div>
            <div className="topbar-divider" aria-hidden="true" />
            <a
              href={`${API_BASE}/docs`}
              target="_blank"
              rel="noopener noreferrer"
              className="topbar-api-link"
              id="topbar-api-docs-link"
              aria-label="Open API documentation"
            >
              <span className="api-link-dot" aria-hidden="true" />
              API v1
            </a>
          </div>
        </header>

        {/* View Router */}
        <main className="workspace-content" id="workspace-content" role="main">
          {activeView === 'dashboard' && <Dashboard onNavigate={setActiveView} />}
          {activeView === 'detect'    && <DetectWorkspace />}
          {activeView === 'recover'   && <RecoverWorkspace />}
          {activeView === 'archive'   && <CaseArchive />}
        </main>
      </div>
    </div>
  );
}
