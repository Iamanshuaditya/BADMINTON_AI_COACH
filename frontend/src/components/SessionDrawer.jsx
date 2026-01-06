import React, { useState } from 'react';
import './SessionDrawer.css';

export default function SessionDrawer({ report, onClose }) {
    const [activeTab, setActiveTab] = useState('summary');

    if (!report) {
        return (
            <div className="drawer empty">
                <div className="drawer-placeholder">
                    <span>Analysis context will appear here</span>
                </div>
            </div>
        );
    }

    // Helper to safely get data
    const events = report.events || [];
    const mistakes = report.top_mistakes || [];
    const fixFirst = report.fix_first_plan;

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
                        {/* Score / High level */}
                        <div className="score-card">
                            <span className="score-label">Events Detected</span>
                            <span className="score-value">{events.length}</span>
                        </div>

                        {/* Fix First Pinned */}
                        {fixFirst && (
                            <div className="drawer-card pinned">
                                <div className="card-label">FIX FIRST</div>
                                <h4>{fixFirst.primary_issue?.replace(/_/g, ' ')}</h4>
                                <p className="drawer-text-sm">{fixFirst.cue}</p>
                                <div className="tag-row">
                                    <span className="tag">{fixFirst.occurrences}x</span>
                                </div>
                            </div>
                        )}

                        {/* Top Issues */}
                        <div className="issues-list">
                            <h4>Top Issues</h4>
                            {mistakes.length === 0 && <p className="drawer-text-sm">Good job! No major issues.</p>}
                            {mistakes.map((m, i) => (
                                <div key={i} className="issue-row">
                                    <span className="issue-name">{m.cue}</span>
                                    <span className="issue-count">{m.count}x</span>
                                </div>
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
                                    <div className="event-conf">Conf: {(ev.confidence * 100).toFixed(0)}%</div>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            <div className="drawer-footer">
                <button className="drawer-action-btn">Export Report</button>
            </div>
        </div>
    );
}
