import React from 'react';
import './TopBar.css';

export default function TopBar({ sessionSummary }) {
  // Default text if no session active
  const sessionLabel = sessionSummary
    ? `${Math.round(sessionSummary.duration || 0)}s session • ${sessionSummary.eventsCount || 0} moves`
    : "Ready for session";

  return (
    <header className="top-bar">
      <div className="logo-section">
        <span className="logo-text">ShuttleSense</span>
      </div>

      <div className="session-pill">
        <span>{sessionLabel}</span>
      </div>

      <div className="settings-actions">
        <button className="new-session-btn">New Session</button>
        <button className="icon-btn">⚙️</button>
      </div>
    </header>
  );
}
