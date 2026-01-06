import { useState, useRef, useCallback } from 'react'
import './App.css'
import TopBar from './components/TopBar'
import ChatInterface from './components/ChatInterface'
import SessionDrawer from './components/SessionDrawer'

// =============================================================================
// Configuration
// =============================================================================

const API_BASE = ''
const API_TIMEOUT = 60000 // 60 seconds for video analysis
const MAX_RETRIES = 3

// =============================================================================
// API Utilities
// =============================================================================

class APIError extends Error {
  constructor(message, code, status, details = {}) {
    super(message)
    this.code = code
    this.status = status
    this.details = details
    this.name = 'APIError'
  }
}

async function fetchWithRetry(url, options = {}, retries = MAX_RETRIES) {
  const timeout = options.timeout || API_TIMEOUT
  let lastError

  for (let attempt = 1; attempt <= retries; attempt++) {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), timeout)

    try {
      const response = await fetch(url, { ...options, signal: controller.signal })
      clearTimeout(timeoutId)

      const contentType = response.headers.get('content-type')
      let data = null
      if (contentType?.includes('application/json')) {
        data = await response.json()
      }

      if (!response.ok) {
        const errorCode = data?.error || 'UNKNOWN_ERROR'
        const errorMessage = data?.detail || `HTTP ${response.status}`
        throw new APIError(errorMessage, errorCode, response.status, data?.details)
      }

      return data

    } catch (error) {
      clearTimeout(timeoutId)
      lastError = error
      if (error.name === 'AbortError') {
        lastError = new APIError('Request timed out', 'TIMEOUT', 408)
      }
      if (error instanceof APIError && error.status >= 400 && error.status < 500) {
        throw error
      }
      if (attempt < retries) {
        await new Promise(r => setTimeout(r, 1000 * attempt))
      }
    }
  }
  throw lastError
}

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
        try { resolve(JSON.parse(xhr.responseText)) }
        catch (e) { resolve(xhr.responseText) }
      } else {
        try {
          const data = JSON.parse(xhr.responseText)
          reject(new APIError(data.detail || `HTTP ${xhr.status}`, data.error || 'UPLOAD_ERROR', xhr.status))
        } catch (e) {
          reject(new APIError(`Upload failed: HTTP ${xhr.status}`, 'UPLOAD_ERROR', xhr.status))
        }
      }
    })

    xhr.addEventListener('error', () => reject(new APIError('Network error', 'NETWORK_ERROR', 0)))
    xhr.addEventListener('timeout', () => reject(new APIError('Timeout', 'TIMEOUT', 408)))

    xhr.timeout = API_TIMEOUT * 2
    xhr.open('POST', url)
    xhr.send(formData)
  })
}

// =============================================================================
// Main App
// =============================================================================

function App() {
  const [file, setFile] = useState(null)
  const [drillType, setDrillType] = useState('unknown')
  const [status, setStatus] = useState('idle')
  const [progress, setProgress] = useState({ stage: '', percent: 0 })
  const [report, setReport] = useState(null)
  const [chatMessages, setChatMessages] = useState([])
  const [chatLoading, setChatLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [error, setError] = useState(null)

  // 1. Upload Handler
  const handleUpload = useCallback(async (selectedFile) => {
    setFile(selectedFile)
    setError(null)
    setStatus('uploading')
    setProgress({ stage: 'Uploading video...', percent: 0 })

    const formData = new FormData()
    formData.append('file', selectedFile)
    formData.append('drill_type', drillType)

    try {
      const result = await uploadWithProgress(
        `${API_BASE}/api/upload-and-analyze`,
        formData,
        (percent) => {
          if (percent < 100) setProgress({ stage: 'Uploading video...', percent })
          else {
            setStatus('analyzing')
            setProgress({ stage: 'Analyzing video... (this may take a minute)', percent: 0 })
          }
        }
      )

      setSessionId(result.session_id)
      setReport(result.report)
      setStatus('complete')

      // Kickoff Message
      setChatMessages([
        {
          role: 'assistant',
          type: 'kickoff',
          data: {
            duration: result.report.video_duration,
            eventsCount: result.report.metrics_summary?.total_events || 0,
            mistakesCount: result.report.metrics_summary?.total_mistakes || 0,
            fixFirst: result.report.fix_first_plan,
            summary: result.report.metrics_summary
          }
        }
      ])

    } catch (err) {
      console.error('Upload failed', err)
      setStatus('error')
      setError(err)
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. Please try again.`,
        error: true
      }])
    }
  }, [drillType])

  // 2. Chat Handler
  const handleSendMessage = useCallback(async (text) => {
    if (!sessionId) return

    setChatMessages(prev => [...prev, { role: 'user', content: text }])
    setChatLoading(true)

    try {
      const result = await fetchWithRetry(
        `${API_BASE}/api/chat`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: sessionId, question: text })
        }
      )

      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: result.answer,
        citations: result.citations,
        grounded: result.grounded
      }])

    } catch (err) {
      setChatMessages(prev => [...prev, {
        role: 'assistant',
        content: "Sorry, I'm having trouble connecting to the coach mind right now.",
        error: true
      }])
    } finally {
      setChatLoading(false)
    }
  }, [sessionId])

  return (
    <div className="app-shell">
      <TopBar sessionSummary={report ? report.metrics_summary : null} />

      <div className="main-layout">
        <div className="chat-columns">
          <ChatInterface
            messages={chatMessages}
            onSendMessage={handleSendMessage}
            onUpload={handleUpload}
            status={status}
            progress={progress}
            drillType={drillType}
            setDrillType={setDrillType}
            chatLoading={chatLoading}
            file={file}
          />

          <SessionDrawer
            report={report}
            onAsk={(question) => handleSendMessage(question)}
          />
        </div>
      </div>
    </div>
  )
}

export default App
