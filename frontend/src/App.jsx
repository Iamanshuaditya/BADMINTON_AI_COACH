import { useState, useRef, useCallback } from 'react'
import './App.css'

// =============================================================================
// Configuration
// =============================================================================

const API_BASE = ''
const API_TIMEOUT = 60000 // 60 seconds for video analysis
const MAX_RETRIES = 3

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

// =============================================================================
// API Utilities with Retry Logic
// =============================================================================

/**
 * Custom API error with code and details
 */
class APIError extends Error {
  constructor(message, code, status, details = {}) {
    super(message)
    this.code = code
    this.status = status
    this.details = details
    this.name = 'APIError'
  }
}

/**
 * Fetch with timeout, retry, and error handling
 */
async function fetchWithRetry(url, options = {}, retries = MAX_RETRIES) {
  const timeout = options.timeout || API_TIMEOUT
  let lastError

  for (let attempt = 1; attempt <= retries; attempt++) {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal
      })

      clearTimeout(timeoutId)

      // Parse response
      const contentType = response.headers.get('content-type')
      let data = null

      if (contentType?.includes('application/json')) {
        data = await response.json()
      }

      // Handle error responses
      if (!response.ok) {
        const errorCode = data?.error || 'UNKNOWN_ERROR'
        const errorMessage = data?.detail || `HTTP ${response.status}`

        // Don't retry client errors (4xx)
        if (response.status >= 400 && response.status < 500) {
          throw new APIError(errorMessage, errorCode, response.status, data?.details)
        }

        throw new APIError(errorMessage, errorCode, response.status, data?.details)
      }

      return data

    } catch (error) {
      clearTimeout(timeoutId)
      lastError = error

      // Handle abort/timeout
      if (error.name === 'AbortError') {
        lastError = new APIError(
          'Request timed out. Please try again.',
          'TIMEOUT',
          408
        )
      }

      // Don't retry on client errors
      if (error instanceof APIError && error.status >= 400 && error.status < 500) {
        throw error
      }

      // Log retry attempt
      if (attempt < retries) {
        console.log(`Retry attempt ${attempt + 1}/${retries} for ${url}`)
        await new Promise(r => setTimeout(r, 1000 * attempt)) // Exponential backoff
      }
    }
  }

  throw lastError
}

/**
 * Upload file with progress tracking
 */
async function uploadWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()

    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        const percent = Math.round((event.loaded / event.total) * 100)
        onProgress(percent)
      }
    })

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText))
        } catch (e) {
          resolve(xhr.responseText)
        }
      } else {
        try {
          const data = JSON.parse(xhr.responseText)
          reject(new APIError(
            data.detail || `HTTP ${xhr.status}`,
            data.error || 'UPLOAD_ERROR',
            xhr.status,
            data.details
          ))
        } catch (e) {
          reject(new APIError(`Upload failed: HTTP ${xhr.status}`, 'UPLOAD_ERROR', xhr.status))
        }
      }
    })

    xhr.addEventListener('error', () => {
      reject(new APIError('Network error during upload', 'NETWORK_ERROR', 0))
    })

    xhr.addEventListener('timeout', () => {
      reject(new APIError('Upload timed out', 'TIMEOUT', 408))
    })

    xhr.timeout = API_TIMEOUT * 2 // Longer timeout for uploads
    xhr.open('POST', url)
    xhr.send(formData)
  })
}

// =============================================================================
// Error Display Component
// =============================================================================

function ErrorDisplay({ error, onDismiss }) {
  if (!error) return null

  const getErrorIcon = (code) => {
    switch (code) {
      case 'RATE_LIMIT_EXCEEDED':
        return '⏱️'
      case 'FILE_TOO_LARGE':
        return '📦'
      case 'INVALID_VIDEO_FORMAT':
        return '🎬'
      case 'INSUFFICIENT_POSE_DATA':
        return '👤'
      case 'TIMEOUT':
        return '⌛'
      default:
        return '⚠️'
    }
  }

  const getErrorHelp = (code) => {
    switch (code) {
      case 'RATE_LIMIT_EXCEEDED':
        return 'Please wait a while before trying again.'
      case 'FILE_TOO_LARGE':
        return 'Try compressing your video or using a shorter clip.'
      case 'INVALID_VIDEO_FORMAT':
        return 'Please use MP4, MOV, or WebM format.'
      case 'INSUFFICIENT_POSE_DATA':
        return 'Make sure your full body is visible and well-lit.'
      case 'TIMEOUT':
        return 'The server is busy. Please try again in a moment.'
      default:
        return null
    }
  }

  return (
    <div className="error-display">
      <div className="error-header">
        <span className="error-icon">{getErrorIcon(error.code)}</span>
        <span className="error-code">{error.code || 'Error'}</span>
        <button className="error-dismiss" onClick={onDismiss}>×</button>
      </div>
      <p className="error-message">{error.message}</p>
      {getErrorHelp(error.code) && (
        <p className="error-help">{getErrorHelp(error.code)}</p>
      )}
    </div>
  )
}

// =============================================================================
// Main App Component
// =============================================================================

function App() {
  // State
  const [file, setFile] = useState(null)
  const [drillType, setDrillType] = useState('unknown')
  const [status, setStatus] = useState('idle') // idle, uploading, analyzing, complete, error
  const [progress, setProgress] = useState({ stage: '', percent: 0 })
  const [error, setError] = useState(null)
  const [report, setReport] = useState(null)
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)

  const fileInputRef = useRef(null)

  // ==========================================================================
  // File Handling
  // ==========================================================================

  const handleFileChange = useCallback((e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      // Validate file type
      const validTypes = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm']
      if (!validTypes.includes(selectedFile.type)) {
        setError(new APIError(
          'Invalid file type. Please upload MP4, MOV, or WebM.',
          'INVALID_VIDEO_FORMAT',
          415
        ))
        return
      }

      // Validate file size (500MB)
      const maxSize = 500 * 1024 * 1024
      if (selectedFile.size > maxSize) {
        setError(new APIError(
          `File too large: ${(selectedFile.size / 1024 / 1024).toFixed(1)}MB (max 500MB)`,
          'FILE_TOO_LARGE',
          413
        ))
        return
      }

      setFile(selectedFile)
      setError(null)
      setReport(null)
      setChatMessages([])
      setSessionId(null)
      setStatus('idle')
    }
  }, [])

  // ==========================================================================
  // Upload & Analyze
  // ==========================================================================

  const handleUploadAndAnalyze = useCallback(async () => {
    if (!file) return

    setError(null)
    setStatus('uploading')
    setProgress({ stage: 'Uploading video...', percent: 0 })

    const formData = new FormData()
    formData.append('file', file)
    formData.append('drill_type', drillType)

    try {
      // Upload with progress
      const result = await uploadWithProgress(
        `${API_BASE}/api/upload-and-analyze`,
        formData,
        (percent) => {
          if (percent < 100) {
            setProgress({ stage: 'Uploading video...', percent })
          } else {
            setStatus('analyzing')
            setProgress({ stage: 'Analyzing video...', percent: 0 })
          }
        }
      )

      // Check for warnings
      if (result.warnings?.length > 0) {
        console.warn('Analysis warnings:', result.warnings)
      }

      setSessionId(result.session_id)
      setReport(result.report)
      setStatus('complete')
      setProgress({ stage: '', percent: 0 })

    } catch (err) {
      console.error('Upload/analyze failed:', err)
      setStatus('error')
      setError(err instanceof APIError ? err : new APIError(err.message, 'UNKNOWN_ERROR', 500))
    }
  }, [file, drillType])

  // ==========================================================================
  // Chat
  // ==========================================================================

  const handleChat = useCallback(async (e) => {
    e.preventDefault()
    if (!chatInput.trim() || !sessionId || chatLoading) return

    const question = chatInput.trim()
    setChatInput('')
    setChatMessages(prev => [...prev, { role: 'user', content: question }])
    setChatLoading(true)

    try {
      const result = await fetchWithRetry(
        `${API_BASE}/api/chat`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, question }),
          timeout: 30000 // 30 second timeout for chat
        },
        2 // Only 2 retries for chat
      )

      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: result.answer,
        citations: result.citations,
        grounded: result.grounded
      }])

    } catch (err) {
      console.error('Chat failed:', err)
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: err instanceof APIError
          ? `Sorry, I couldn't process that: ${err.message}`
          : 'Sorry, something went wrong. Please try again.',
        error: true
      }])
    } finally {
      setChatLoading(false)
    }
  }, [chatInput, sessionId, chatLoading])

  // ==========================================================================
  // Render
  // ==========================================================================

  return (
    <div className="app">
      <header className="header">
        <h1>🏸 ShuttleSense</h1>
        <p>AI-Powered Badminton Footwork Coach</p>
      </header>

      <main className="main">
        {/* Error Display */}
        <ErrorDisplay error={error} onDismiss={() => setError(null)} />

        {/* Upload Section */}
        <section className="upload-section">
          <h2>Upload Video</h2>

          <div
            className={`upload-area ${file ? 'has-file' : ''}`}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            {file ? (
              <div className="file-info">
                <span className="file-icon">🎬</span>
                <span className="file-name">{file.name}</span>
                <span className="file-size">({(file.size / 1024 / 1024).toFixed(1)} MB)</span>
              </div>
            ) : (
              <div className="upload-prompt">
                <span className="upload-icon">📁</span>
                <span>Click to select video</span>
                <span className="hint">MP4, MOV, or WebM • Max 500MB</span>
              </div>
            )}
          </div>

          <div className="controls">
            <select
              value={drillType}
              onChange={(e) => setDrillType(e.target.value)}
              disabled={status === 'uploading' || status === 'analyzing'}
            >
              {DRILL_TYPES.map(d => (
                <option key={d.value} value={d.value}>{d.label}</option>
              ))}
            </select>

            <button
              className="analyze-btn"
              onClick={handleUploadAndAnalyze}
              disabled={!file || status === 'uploading' || status === 'analyzing'}
            >
              {status === 'uploading' ? 'Uploading...' :
                status === 'analyzing' ? 'Analyzing...' :
                  'Analyze Video'}
            </button>
          </div>

          {/* Progress Bar */}
          {(status === 'uploading' || status === 'analyzing') && (
            <div className="progress-container">
              <div className="progress-text">{progress.stage}</div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{
                    width: status === 'analyzing' ? '100%' : `${progress.percent}%`,
                    animation: status === 'analyzing' ? 'pulse 2s infinite' : 'none'
                  }}
                />
              </div>
            </div>
          )}
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
                      <span className="event-time">[{e.timestamp?.toFixed(1)}s]</span>
                      <span className="event-type">{e.type?.replace(/_/g, ' ')}</span>
                      <span className="event-conf">{((e.confidence || 0) * 100).toFixed(0)}%</span>
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
                <div key={i} className={`chat-message ${msg.role} ${msg.error ? 'error' : ''}`}>
                  <div className="message-content">{msg.content}</div>
                  {msg.citations?.length > 0 && (
                    <div className="citations">
                      Citations: {msg.citations.join(', ')}
                    </div>
                  )}
                </div>
              ))}
              {chatLoading && (
                <div className="chat-message assistant loading">
                  <div className="typing-indicator">
                    <span></span><span></span><span></span>
                  </div>
                </div>
              )}
            </div>

            <form className="chat-input-form" onSubmit={handleChat}>
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Ask about your footwork..."
                disabled={chatLoading}
              />
              <button type="submit" disabled={!chatInput.trim() || chatLoading}>
                {chatLoading ? '...' : 'Send'}
              </button>
            </form>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>ShuttleSense V2 • Built for improvement, not perfection</p>
      </footer>
    </div>
  )
}

export default App
