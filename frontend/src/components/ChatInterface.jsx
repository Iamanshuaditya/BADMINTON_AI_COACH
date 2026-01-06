import React, { useRef, useEffect, useMemo, useState } from 'react';
import './ChatInterface.css';

export default function ChatInterface({
    messages,
    onSendMessage,
    onUpload,
    status,
    progress,
    drillType,
    setDrillType,
    chatLoading,
    file
}) {
    const bottomRef = useRef(null);
    const inputRef = useRef(null);
    const fileInputRef = useRef(null);

    // Video URL
    const videoUrl = useMemo(() => {
        return file ? URL.createObjectURL(file) : null;
    }, [file]);

    useEffect(() => {
        return () => {
            if (videoUrl) URL.revokeObjectURL(videoUrl);
        };
    }, [videoUrl]);

    // Auto-scroll
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, status, progress, chatLoading]);

    // Input placeholder logic
    const getPlaceholder = () => {
        if (status !== 'complete') return "Waiting for analysis...";
        // Check if last message was the kickoff or an answer
        const lastMsg = messages[messages.length - 1];
        if (lastMsg?.type === 'kickoff') {
            const issue = lastMsg.data?.fixFirst?.primary_issue?.replace(/_/g, ' ') || 'it';
            return `Ask how to fix ${issue}...`;
        }
        return "Ask about your footwork...";
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const text = inputRef.current.value.trim();
            if (text) {
                onSendMessage(text);
                inputRef.current.value = '';
            }
        }
    };

    const handleFileSelect = (e) => {
        if (e.target.files?.[0]) {
            onUpload(e.target.files[0]);
        }
    };

    return (
        <div className="chat-container">
            <div className="chat-messages-area">
                {/* Upload Hero */}
                {status === 'idle' && messages.length === 0 && (
                    <div className="upload-hero fade-in">
                        <h2 className="hero-title">ShuttleSense Coach</h2>
                        <p className="hero-subtitle">Upload a practice video to get a professional break-down.</p>

                        <div
                            className="drop-zone"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <span className="drop-icon">⚡</span>
                            <span className="drop-text">Drop video here</span>
                            <span className="drop-hint">Fixed tripod • Full body • 30–120s</span>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="video/*"
                                className="hidden-input"
                                onChange={handleFileSelect}
                            />
                        </div>

                        <div className="drill-selector-wrapper">
                            <select
                                value={drillType}
                                onChange={(e) => setDrillType(e.target.value)}
                                className="drill-dropdown"
                            >
                                <option value="unknown">Select drill type...</option>
                                <option value="6-corner-shadow">6-Corner Shadow Footwork</option>
                                <option value="side-to-side">Side-to-Side Defensive</option>
                                <option value="front-back">Front-Back Movement</option>
                                <option value="overhead-shadow">Overhead Shadow (Clear/Smash)</option>
                            </select>
                        </div>
                    </div>
                )}

                {/* Progress */}
                {(status === 'uploading' || status === 'analyzing') && (
                    <div className="system-note fade-in">
                        <div className="system-note-content">
                            <div className="loader-pulse"></div>
                            <span>{progress.stage}</span>
                        </div>
                        {status === 'uploading' && (
                            <div className="minimal-progress">
                                <div className="minimal-fill" style={{ width: `${progress.percent}%` }}></div>
                            </div>
                        )}
                    </div>
                )}

                {/* Messages */}
                {messages.map((msg, idx) => (
                    <MessageItem key={idx} message={msg} onAction={(action) => onSendMessage(action)} videoUrl={videoUrl} />
                ))}

                {chatLoading && (
                    <div className="message assistant loading fade-in">
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                    </div>
                )}

                <div ref={bottomRef} />
            </div>

            {/* Input Area */}
            <div className="chat-input-area">
                {/* Mini Player anchored here if active */}
                {status === 'complete' && videoUrl && (
                    <div className="mini-player-anchor">
                        {/* Logic to show player could be here, but for now we keep it optional/inline */}
                    </div>
                )}

                <div className="input-wrapper">
                    <button className="attach-btn" disabled={status !== 'complete' && status !== 'idle'}>
                        +
                    </button>
                    <textarea
                        ref={inputRef}
                        placeholder={getPlaceholder()}
                        rows={1}
                        onKeyDown={handleKeyDown}
                        disabled={status !== 'complete' && status !== 'idle'}
                    />
                    <button
                        className="send-btn"
                        onClick={() => {
                            const text = inputRef.current.value.trim();
                            if (text) {
                                onSendMessage(text);
                                inputRef.current.value = '';
                            }
                        }}
                        disabled={status !== 'complete' && status !== 'idle'}
                    >
                        →
                    </button>
                </div>
            </div>
        </div>
    );
}

function MessageItem({ message, onAction, videoUrl }) {
    const isUser = message.role === 'user';

    if (message.type === 'kickoff') {
        return <KickoffCard data={message.data} onAction={onAction} />;
    }

    // User Message
    if (isUser) {
        return (
            <div className="message user fade-in">
                <div className="message-content">{message.content}</div>
            </div>
        );
    }

    // Coach Message (Standard)
    return (
        <div className="message assistant fade-in">
            <div className="coach-avatar">C</div>
            <div className="message-body">
                <div className="message-content">
                    {formatCoachText(message.content)}
                </div>

                {/* Evidence Block (Collapsible logic could be added here, for now it's "On Demand" via the button usually, but if provided we show it elegantly) */}
                {message.citations && message.citations.length > 0 && (
                    <EvidenceBlock citations={message.citations} />
                )}
            </div>
        </div>
    );
}

function KickoffCard({ data, onAction }) {
    const { fixFirst } = data;
    if (!fixFirst) return null;

    const issueName = fixFirst.primary_issue?.replace(/_/g, ' ');

    return (
        <div className="kickoff-container fade-in">
            <div className="verdict-card">
                <div className="verdict-header">
                    <span className="verdict-label">FIX FIRST</span>
                    <h1 className="verdict-title">{issueName}</h1>
                </div>

                <div className="verdict-body">
                    <p className="verdict-text">
                        This is the main issue holding your recovery back.
                        Detected repeatedly across the session.
                    </p>

                    <div className="verdict-cue-box">
                        <span className="cue-label">PRIMARY CUE</span>
                        <p className="cue-text">"{fixFirst.cue}"</p>
                    </div>

                    <div className="verdict-actions">
                        <button
                            className="action-btn primary"
                            // In a real app this would trigger the evidence view state
                            onClick={() => onAction(`Show me evidence of ${issueName}`)}
                        >
                            Show evidence
                        </button>
                        <button
                            className="action-btn secondary"
                            onClick={() => onAction(`How do I fix ${issueName}?`)}
                        >
                            Give me the drill
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

function EvidenceBlock({ citations }) {
    const [expanded, setExpanded] = useState(false);

    if (!expanded) {
        return (
            <button className="evidence-toggle" onClick={() => setExpanded(true)}>
                <span className="toggle-icon">▶</span> Show Session Evidence
            </button>
        );
    }

    return (
        <div className="evidence-block fade-in">
            <div className="evidence-header" onClick={() => setExpanded(false)}>
                <span className="toggle-icon">▼</span> Evidence from your session
            </div>
            <div className="evidence-pills">
                {citations.map((time, i) => (
                    <button key={i} className="timestamp-pill">
                        {time}s
                    </button>
                ))}
            </div>
        </div>
    );
}

// Helper to add line breaks to coach answers
function formatCoachText(text) {
    return text.split('\n').map((line, i) => (
        <p key={i} className={line.trim() === '' ? 'spacer' : 'text-line'}>
            {line}
        </p>
    ));
}
