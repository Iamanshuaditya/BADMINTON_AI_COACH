import React from 'react';
import './TopBar.css';

export default function TopBar() {
  return (
    <header className="top-bar">
      <div className="logo-section">
        <span className="logo-icon">🏸</span>
        <span className="logo-text">ShuttleSense</span>
      </div>
      
      <div className="session-selector">
        {/* Placeholder for session selector */}
        <span className="current-session">New Session</span>
      </div>
      
      <div className="settings-actions">
        <button className="icon-btn">⚙️</button>
      </div>
    </header>
  );
}
