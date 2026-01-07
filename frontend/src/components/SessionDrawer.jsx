import React, { useState } from 'react';
import './SessionDrawer.css';

export default function SessionDrawer({ report, onAsk }) {
    const [activeTab, setActiveTab] = useState('summary');

    if (!report) {
        return (
            <div className="drawer empty">
                <div className="drawer-placeholder">
                    <span>Session context</span>
                </div>
            </div>
        );
    }

    const events = report.events || [];
    const mistakes = report.top_mistakes || [];
    const fixFirst = report.fix_first_plan;
    const drillType = report.drill_type || 'unknown';
    const drillSource = report.drill_type_source || 'user';
    const drillConfidence = typeof report.drill_type_confidence === 'number'
        ? report.drill_type_confidence.toFixed(2)
        : '1.00';

    return (
        <div className="drawer">
            <div className="drawer-header">
                <h3>Session Context</h3>
            </div>

            <div className="drawer-tabs">
                <button
                    className={`tab-btn ${activeTab === 'summary' ? 'active' : ''}`}
                    onClick={() => setActiveTab('summary')}
                >
                    Summary
                </button>
                <button
                    className={`tab-btn ${activeTab === 'timeline' ? 'active' : ''}`}
                    onClick={() => setActiveTab('timeline')}
                >
                    Timeline
                </button>
            </div>

            <div className="drawer-content">
                {activeTab === 'summary' && (
                    <div className="section fade-in">
                        <div className="text-summary">
                            <p className="summary-line">{events.length} movement events analyzed</p>
                            <p className="summary-line">{mistakes.reduce((acc, m) => acc + m.count, 0)} form breakdowns detected</p>
                            <p className="summary-line">Drill: {drillType} ({drillSource}, {drillConfidence})</p>
                        </div>

                        {fixFirst && (
                            <div className="drawer-card pinned">
                                <div className="card-label">FIX FIRST</div>
                                <h4>{fixFirst.primary_issue?.replace(/_/g, ' ')}</h4>
                                <div className="drawer-card-actions">
                                    <button className="drawer-link" onClick={() => onAsk(`Tell me more about ${fixFirst.primary_issue}`)}>
                                        Explain this →
                                    </button>
                                </div>
                            </div>
                        )}

                        <div className="issues-list">
                            <h4>Top Issues</h4>
                            {mistakes.length === 0 && <p className="drawer-text-sm">No major issues found.</p>}
                            {mistakes.map((m, i) => (
                                <button
                                    key={i}
                                    className="issue-row-btn"
                                    onClick={() => onAsk(`What about my ${m.cue}?`)}
                                >
                                    <span className="issue-name">{m.cue}</span>
                                    <span className="issue-count">{m.count} times</span>
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {activeTab === 'timeline' && (
                    <div className="timeline-list fade-in">
                        {events.map((ev, i) => (
                            <div key={i} className="timeline-item">
                                <div className="time-marker">{ev.timestamp?.toFixed(1)}s</div>
                                <div className="event-info">
                                    <div className="event-name">{ev.type?.replace(/_/g, ' ')}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}
