import React, { useState } from 'react';

const NAV = [
  { id: 'dashboard', label: 'Overview',  sub: 'Home',             icon: 'grid'    },
  { id: 'detect',    label: 'Detect',    sub: 'Forensic Analysis', icon: 'search'  },
  { id: 'recover',   label: 'Recover',   sub: 'AI Reconstruction', icon: 'refresh' },
  { id: 'archive',   label: 'Archive',   sub: 'Case Archive',      icon: 'archive' },
];

function GridIcon()    { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/></svg>; }
function SearchIcon()  { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>; }
function RefreshIcon() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>; }
function ArchiveIcon() { return <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 7V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v2"/><line x1="12" y1="12" x2="12" y2="16"/><line x1="10" y1="14" x2="14" y2="14"/></svg>; }
const ICON_MAP = { grid: <GridIcon/>, search: <SearchIcon/>, refresh: <RefreshIcon/>, archive: <ArchiveIcon/> };

export default function Sidebar({ activeView, onNavigate }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside className={`sidebar${collapsed ? ' sidebar-collapsed' : ''}`} aria-label="Main navigation">

      {/* Logo */}
      <div className="sidebar-logo" onClick={() => onNavigate('dashboard')} role="button" tabIndex={0}
        onKeyDown={e => e.key === 'Enter' && onNavigate('dashboard')}>
        <div className="sidebar-logo-icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--cyan)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
        </div>
        {!collapsed && (
          <div className="sidebar-logo-text">
            <span className="sidebar-logo-name">PhantomaShield</span>
            <span className="sidebar-logo-sub">Medical AI Platform</span>
          </div>
        )}
      </div>

      {/* Collapse toggle */}
      <button
        className="sidebar-collapse-btn"
        onClick={() => setCollapsed(c => !c)}
        aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
          {collapsed
            ? <polyline points="9 18 15 12 9 6"/>
            : <polyline points="15 18 9 12 15 6"/>}
        </svg>
      </button>

      {/* Nav */}
      <nav className="sidebar-nav" role="navigation">
        {!collapsed && <div className="sidebar-section-label">WORKSPACE</div>}
        {NAV.map(item => (
          <button
            key={item.id}
            className={`sidebar-nav-item${
              activeView === item.id ? ' active' : ''
            }${
              item.id === 'recover' ? ' accent-purple' :
              item.id === 'archive' ? ' accent-amber'  : ''
            }`}
            onClick={() => onNavigate(item.id)}
            aria-current={activeView === item.id ? 'page' : undefined}
            title={collapsed ? item.label : undefined}
            id={`nav-${item.id}`}
          >
            <span className="nav-item-icon" aria-hidden="true">{ICON_MAP[item.icon]}</span>
            {!collapsed && (
              <div className="nav-item-content">
                <span className="nav-item-label">{item.label}</span>
                <span className="nav-item-sub">{item.sub}</span>
              </div>
            )}
            {activeView === item.id && <span className="nav-item-indicator" aria-hidden="true"/>}
          </button>
        ))}
      </nav>

      {/* Divider + Status */}
      <div className="sidebar-divider"/>
      {!collapsed && (
        <div className="sb-status">
          <div className="sb-status-row">
            <span className="sb-dot" aria-hidden="true"/>
            <span className="sb-status-text">All Systems Operational</span>
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="sidebar-footer">
        {!collapsed && <div className="sidebar-version">v2.0 · Clinical-Grade</div>}
      </div>
    </aside>
  );
}
