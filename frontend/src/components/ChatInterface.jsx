import React, { useRef, useEffect, useMemo } from 'react';
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

    // Create video URL safely
    const videoUrl = useMemo(() => {
        return file ? URL.createObjectURL(file) : null;
    }, [file]);

    // Clean up URL on unmount
    useEffect(() => {
        return () => {
            if (videoUrl) URL.revokeObjectURL(videoUrl);
        };
    }, [videoUrl]);

    // Auto-scroll to bottom
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, status, progress]);

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
                {/* Empty State / Upload */}
                {status === 'idle' && messages.length === 0 && (
                    <div className="upload-hero fade-in">
                        <h2>Drop your practice video</h2>
                        <p>Best results: fixed tripod, full body visible, 30–180s.</p>

                        <div
                            className="drop-zone"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <span className="drop-icon">☁️</span>
                            <span>Upload Video</span>
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="video/*"
                                className="hidden-input"
                                onChange={handleFileSelect}
                            />
                        </div>

                        <div className="drill-selector">
                            <select
                                value={drillType}
                                onChange={(e) => setDrillType(e.target.value)}
                                className="drill-dropdown"
                            >
                                <option value="unknown">Select drill type...</option>
                                <option value="6-corner-shadow">🦶 6-Corner Shadow Footwork</option>
                                <option value="side-to-side">🦶 Side-to-Side Defensive</option>
                                <option value="front-back">🦶 Front-Back Movement</option>
                                <option value="overhead-shadow">🏸 Overhead Shadow (Clear/Smash)</option>
                                <option value="overhead-clear">🏸 Clear Shadow Practice</option>
                                <option value="overhead-smash">🏸 Smash Shadow Practice</option>
                            </select>
                        </div>
                    </div>
                )}

                {/* Progress State */}
                {(status === 'uploading' || status === 'analyzing') && (
                    <div className="system-message progress-message fade-in">
                        <div className="loader-spinner"></div>
                        <div className="progress-content">
                            <span className="progress-stage">{progress.stage}</span>
                            {status === 'uploading' && (
                                <div className="progress-track">
                                    <div className="progress-fill" style={{ width: `${progress.percent}%` }}></div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Message Stream */}
                {messages.map((msg, idx) => (
                    <MessageItem key={idx} message={msg} />
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

            {/* Mini Player */}
            {status === 'complete' && videoUrl && (
                <div className="mini-player-strip fade-in">
                    <video src={videoUrl} controls className="mini-video" />
                    <div className="player-hint">Reviewing your session</div>
                </div>
            )}

            {/* Input Area */}
            <div className="chat-input-area">
                <div className="input-wrapper">
                    <button className="attach-btn" disabled={status !== 'complete' && status !== 'idle'}>
                        📎
                    </button>
                    <textarea
                        ref={inputRef}
                        placeholder={status === 'complete' ? "Ask about your footwork..." : "Waiting for analysis..."}
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
                        ➤
                    </button>
                </div>
            </div>
        </div>
    );
}

function MessageItem({ message }) {
    const isUser = message.role === 'user';

    if (message.type === 'kickoff') {
        return <KickoffCard data={message.data} />;
    }

    return (
        <div className={`message ${isUser ? 'user' : 'assistant'} fade-in`}>
            <div className="message-content">
                {message.content}
            </div>
            {message.citations && message.citations.length > 0 && (
                <div className="evidence-block">
                    <span className="evidence-label">Evidence:</span>
                    {message.citations.map((cit, i) => (
                        <span key={i} className="evidence-pill">{cit}</span>
                    ))}
                </div>
            )}
        </div>
    );
}

function KickoffCard({ data }) {
    const { summary, fixFirst } = data;
    return (
        <div className="kickoff-container fade-in">
            <div className="summary-chips">
                <div className="chip">⏱ {data.duration?.toFixed(1)}s</div>
                <div className="chip">⚡ {data.eventsCount} Events</div>
                <div className="chip">⚠️ {data.mistakesCount} Issues</div>
            </div>

            {fixFirst && (
                <div className="fix-first-card">
                    <div className="card-header">
                        <span className="badge-warning">FIX FIRST</span>
                        <h4>{fixFirst.primary_issue?.replace(/_/g, ' ')}</h4>
                    </div>
                    <p className="fix-reason">
                        {fixFirst.cue}
                    </p>
                    <div className="fix-stats">
                        <span>{fixFirst.occurrences} occurrences</span>
                        <span>Focus: {fixFirst.focus_drill}</span>
                    </div>
                </div>
            )}
        </div>
    );
}
