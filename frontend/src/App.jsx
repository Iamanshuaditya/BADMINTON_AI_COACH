import { useState, useRef } from 'react'
import './App.css'

const API_BASE = ''

const DRILL_TYPES = [
  { value: 'unknown', label: 'Select drill type...' },
  // Footwork drills
  { value: '6-corner-shadow', label: '🦶 6-Corner Shadow Footwork' },
  { value: 'side-to-side', label: '🦶 Side-to-Side Defensive' },
  { value: 'front-back', label: '🦶 Front-Back Movement' },
  // Stroke drills
  { value: 'overhead-shadow', label: '🏸 Overhead Shadow (Clear/Smash)' },
  { value: 'overhead-clear', label: '🏸 Clear Shadow Practice' },
  { value: 'overhead-smash', label: '🏸 Smash Shadow Practice' },
]

function App() {
  const [file, setFile] = useState(null)
  const [drillType, setDrillType] = useState('unknown')
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState('')
  const [report, setReport] = useState(null)
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [sessionId, setSessionId] = useState(null)
  const fileInputRef = useRef(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setReport(null)
      setChatMessages([])
      setSessionId(null)
    }
  }

  const handleUploadAndAnalyze = async () => {
    if (!file) return

    setStatus('uploading')
    setProgress('Uploading video...')

    const formData = new FormData()
    formData.append('file', file)
    formData.append('drill_type', drillType)

    try {
      const response = await fetch(`${API_BASE}/api/upload-and-analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || 'Analysis failed')
      }

      setProgress('Analyzing video...')
      const result = await response.json()

      setSessionId(result.session_id)
      setReport(result.report)
      setStatus('complete')
      setProgress('')
    } catch (error) {
      setStatus('error')
      setProgress(`Error: ${error.message}`)
    }
  }

  const handleChat = async (e) => {
    e.preventDefault()
    if (!chatInput.trim() || !sessionId) return

    const question = chatInput.trim()
    setChatInput('')
    setChatMessages(prev => [...prev, { role: 'user', content: question }])

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, question }),
      })

      const result = await response.json()
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: result.answer,
        citations: result.citations,
        grounded: result.grounded
      }])
    } catch (error) {
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${error.message}`,
        grounded: false
      }])
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🏸 ShuttleSense</h1>
        <p>AI-Powered Badminton Footwork Coach</p>
      </header>

      <main className="main">
        {/* Upload Section */}
        <section className="upload-section">
          <h2>Upload Video</h2>

          <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            {file ? (
              <div className="file-info">
                <span className="file-icon">🎬</span>
                <span>{file.name}</span>
                <span className="file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
              </div>
            ) : (
              <div className="upload-prompt">
                <span className="upload-icon">📁</span>
                <span>Click to select video</span>
                <span className="hint">MP4, MOV, or WebM</span>
              </div>
            )}
          </div>

          <div className="controls">
            <select value={drillType} onChange={(e) => setDrillType(e.target.value)}>
              {DRILL_TYPES.map(d => (
                <option key={d.value} value={d.value}>{d.label}</option>
              ))}
            </select>

            <button
              className="analyze-btn"
              onClick={handleUploadAndAnalyze}
              disabled={!file || status === 'uploading'}
            >
              {status === 'uploading' ? 'Analyzing...' : 'Analyze Video'}
            </button>
          </div>

          {progress && <div className="progress">{progress}</div>}
        </section>

        {/* Report Section */}
        {report && (
          <section className="report-section">
            <h2>Coach Report</h2>

            <div className="report-grid">
              {/* Summary Card */}
              <div className="card summary-card">
                <h3>Session Summary</h3>
                <div className="stats">
                  <div className="stat">
                    <span className="stat-value">{report.video_duration?.toFixed(1)}s</span>
                    <span className="stat-label">Duration</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{report.metrics_summary?.total_events || 0}</span>
                    <span className="stat-label">Events</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{report.metrics_summary?.total_mistakes || 0}</span>
                    <span className="stat-label">Issues</span>
                  </div>
                </div>
              </div>

              {/* Fix First Plan */}
              {report.fix_first_plan && (
                <div className="card fix-card">
                  <h3>🎯 Fix First</h3>
                  <div className="fix-content">
                    <div className="fix-cue">{report.fix_first_plan.cue}</div>
                    <div className="fix-issue">{report.fix_first_plan.primary_issue?.replace(/_/g, ' ')}</div>
                    <div className="fix-occurrences">{report.fix_first_plan.occurrences} occurrences</div>
                    <div className="fix-drill">{report.fix_first_plan.focus_drill}</div>
                  </div>
                </div>
              )}

              {/* Top Mistakes */}
              <div className="card mistakes-card">
                <h3>Top Issues</h3>
                {report.top_mistakes?.length > 0 ? (
                  <ul className="mistakes-list">
                    {report.top_mistakes.map((m, i) => (
                      <li key={i} className="mistake-item">
                        <span className="mistake-cue">{m.cue}</span>
                        <span className="mistake-count">{m.count}x</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="no-issues">No major issues detected! 🎉</p>
                )}
              </div>

              {/* Events Timeline */}
              <div className="card events-card">
                <h3>Events Timeline</h3>
                <div className="events-list">
                  {report.events?.slice(0, 10).map((e, i) => (
                    <div key={i} className="event-item">
                      <span className="event-time">[{e.timestamp}s]</span>
                      <span className="event-type">{e.type?.replace(/_/g, ' ')}</span>
                      <span className="event-conf">{(e.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                  {report.events?.length > 10 && (
                    <div className="more-events">+ {report.events.length - 10} more events</div>
                  )}
                </div>
              </div>
            </div>

            {/* Raw JSON Toggle */}
            <details className="json-details">
              <summary>View Raw JSON</summary>
              <pre className="json-view">{JSON.stringify(report, null, 2)}</pre>
            </details>
          </section>
        )}

        {/* Chat Section */}
        {sessionId && (
          <section className="chat-section">
            <h2>Ask the Coach</h2>

            <div className="chat-messages">
              {chatMessages.length === 0 && (
                <div className="chat-hint">
                  Ask questions about your session. Examples:
                  <ul>
                    <li>"What was my biggest issue?"</li>
                    <li>"Show me when my knee collapsed"</li>
                    <li>"How was my split step timing?"</li>
                  </ul>
                </div>
              )}
              {chatMessages.map((msg, i) => (
                <div key={i} className={`chat-message ${msg.role}`}>
                  <div className="message-content">{msg.content}</div>
                  {msg.citations?.length > 0 && (
                    <div className="citations">
                      Citations: {msg.citations.join(', ')}
                    </div>
                  )}
                </div>
              ))}
            </div>

            <form className="chat-input-form" onSubmit={handleChat}>
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about your footwork..."
              />
              <button type="submit" disabled={!chatInput.trim()}>Send</button>
            </form>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>ShuttleSense V1 • Built for improvement, not perfection</p>
      </footer>
    </div>
  )
}

export default App
