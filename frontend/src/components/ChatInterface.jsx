import React, { useState } from 'react';
import './ChatInterface.css';

const DRILL_OPTIONS = [
  { value: 'unknown', label: 'Auto-detect' },
  { value: 'footwork', label: 'Footwork' },
  { value: '6-corner-shadow', label: '6-corner shadow' },
  { value: 'overhead-shadow', label: 'Overhead shadow' }
];

export default function ChatInterface({
  messages,
  onSendMessage,
  onUpload,
  status,
  progress,
  drillType,
  setDrillType,
  chatLoading
}) {
  const [draft, setDraft] = useState('');

  const handleSend = () => {
    const text = draft.trim();
    if (!text) return;
    onSendMessage(text);
    setDraft('');
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file) {
      onUpload(file);
    }
  };

  return (
    <div className="chat-panel">
      <div className="upload-bar">
        <label className="upload-btn">
          <input type="file" accept="video/*" onChange={handleFileChange} />
          Upload video
        </label>
        <select
          className="drill-select"
          value={drillType}
          onChange={(e) => setDrillType(e.target.value)}
        >
          {DRILL_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <div className="upload-status">
          {status === 'uploading' && `${progress.stage} ${progress.percent}%`}
          {status === 'analyzing' && progress.stage}
          {status === 'complete' && 'Analysis complete'}
        </div>
      </div>

      <div className="chat-messages">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`chat-message ${msg.role === 'user' ? 'user' : 'assistant'}`}
          >
            {msg.type === 'kickoff' && msg.data ? (
              <div className="kickoff-card">
                <div className="kickoff-title">Session summary</div>
                <div>{Math.round(msg.data.duration || 0)}s video</div>
                <div>{msg.data.eventsCount || 0} events, {msg.data.mistakesCount || 0} mistakes</div>
                {msg.data.fixFirst?.primary_issue && (
                  <div>Fix first: {msg.data.fixFirst.primary_issue.replace(/_/g, ' ')}</div>
                )}
              </div>
            ) : (
              <div className="message-body">
                <div>{msg.content}</div>
                {msg.grounded === false && (
                  <div className="grounded-note">
                    Not enough evidence for this question.
                    {msg.missing_evidence?.reason ? ` (${msg.missing_evidence.reason})` : ''}
                  </div>
                )}
                {msg.citations?.length ? (
                  <div className="citations">
                    {msg.citations.map((c) => `[${c.timestamp}s]`).join(' ')}
                  </div>
                ) : null}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="chat-input">
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask about your session..."
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          disabled={chatLoading}
        />
        <button onClick={handleSend} disabled={chatLoading || !draft.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
