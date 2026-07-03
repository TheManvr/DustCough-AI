import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { copy } from '../assets/copy'
import { API_BASE_URL } from '../config/api'
import { logos } from '../assets/logos'
import './RecordPage.css'

const WAVEFORM_BAR_COUNT = 18
const NOISE_CALIBRATION_SECONDS = 0.8
const PRE_ROLL_SECONDS = 1.0
const POST_ROLL_SECONDS = 2.8
const MIN_TRIGGER_RMS = 0.012
const MIN_TRIGGER_PEAK = 0.045
const TRIGGER_MULTIPLIER = 3.2
const MIN_BURST_SECONDS = 0.055
const MAX_BURST_SECONDS = 1.15
const ACCEPTED_COUGH_CONFIDENCE = 0.7
const PREDICT_TIMEOUT_MS = 30_000
const PREDICT_TIMEOUT_MESSAGE = 'การวิเคราะห์ใช้เวลานานเกินไป อาจเกิดจากเซิร์ฟเวอร์กำลังเริ่มทำงาน กรุณาลองใหม่อีกครั้ง'
const IDLE_WAVEFORM_LEVELS = Array.from(
  { length: WAVEFORM_BAR_COUNT },
  (_, index) => 0.24 + (index % 5) * 0.07,
)

const STATE_TEXT = {
  idle: 'กดเริ่มตรวจจับ แล้วไอ 1–2 ครั้ง ระบบจะรอฟังจนกว่าจะเจอเสียงไอที่ชัดพอ',
  listening: 'กำลังตรวจจับเสียงไอ... ไอได้เลยเมื่อพร้อม',
  cough_candidate_detected: 'ตรวจพบช่วงเสียงที่คล้ายเสียงไอ กำลังจับช่วงเสียง...',
  capturing: 'กำลังเก็บช่วงเสียงไอให้ครบถ้วน...',
  analyzing: 'กำลังให้ AI ตรวจว่าเป็นเสียงไอชัดพอหรือไม่...',
  retry_listening: 'เสียงยังไม่ชัดพอ กรุณาไออีกครั้ง',
  accepted: 'AI ยืนยันเสียงไอชัดเจน กำลังไปขั้นตอนถัดไป...',
  too_quiet: 'เสียงเบาเกินไป กรุณาขยับเข้าใกล้ไมโครโฟนมากขึ้น',
  error: 'ไม่สามารถเริ่มตรวจจับเสียงไอได้',
}

function floatTo16BitPcm(view, offset, input) {
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]))
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
    offset += 2
  }
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i += 1) {
    view.setUint8(offset + i, value.charCodeAt(i))
  }
}

function pcmToWavBlob(samples, sampleRate) {
  const bytesPerSample = 2
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample)
  const view = new DataView(buffer)

  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + samples.length * bytesPerSample, true)
  writeString(view, 8, 'WAVE')
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * bytesPerSample, true)
  view.setUint16(32, bytesPerSample, true)
  view.setUint16(34, 16, true)
  writeString(view, 36, 'data')
  view.setUint32(40, samples.length * bytesPerSample, true)
  floatTo16BitPcm(view, 44, samples)

  return new Blob([buffer], { type: 'audio/wav' })
}

function mergeChunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const merged = new Float32Array(totalLength)
  let offset = 0

  chunks.forEach((chunk) => {
    merged.set(chunk, offset)
    offset += chunk.length
  })

  return merged
}

function getAudioFeatures(samples) {
  let squareSum = 0
  let peak = 0

  for (let i = 0; i < samples.length; i += 1) {
    const value = samples[i]
    const absValue = Math.abs(value)
    squareSum += value * value
    if (absValue > peak) peak = absValue
  }

  return {
    rms: Math.sqrt(squareSum / Math.max(1, samples.length)),
    peak,
  }
}

function isAcceptedCough(aiResult) {
  return aiResult?.label === 'cough' && Number(aiResult.confidence || 0) >= ACCEPTED_COUGH_CONFIDENCE
}

function normalizeBackendLabel(label) {
  if (label === 'noise' || label === 'unknown' || !label) return 'unclear'
  return label
}

function getRejectedAnalysisMessage(aiResult) {
  const confidencePercent = Math.round((Number(aiResult.confidence) || 0) * 100)

  switch (aiResult.label) {
    case 'cough':
      return `ตรวจพบเสียงไอ แต่ความมั่นใจยังไม่ถึง 70% (${confidencePercent}%) กรุณาลองตรวจจับใหม่`
    case 'non_cough':
      return 'ยังไม่พบเสียงไอที่ชัดเจน กรุณากดลองตรวจจับใหม่แล้วไออีกครั้ง'
    case 'unclear':
      return 'พบเสียงที่ยังไม่ชัดเจน กรุณาลองใหม่ในที่เงียบขึ้นและไอใกล้ไมโครโฟนมากขึ้น'
    case 'too_quiet':
      return 'เสียงเบาเกินไป กรุณาขยับเข้าใกล้ไมโครโฟนมากขึ้น'
    case 'too_short':
      return 'เสียงสั้นเกินไป กรุณาไอ 1–2 ครั้งให้ชัดเจนแล้วลองตรวจจับใหม่'
    case 'uncertain_cough':
      return `พบรูปแบบเสียงที่คล้ายเสียงไอ แต่ความมั่นใจยังไม่สูง (${confidencePercent}%) กรุณาลองตรวจจับใหม่`
    default:
      return 'ระบบยังไม่สามารถยืนยันเสียงไอได้ กรุณาลองตรวจจับใหม่อีกครั้ง'
  }
}

function RecordPage() {
  const navigate = useNavigate()

  const [captureState, setCaptureState] = useState('idle')
  const [permError, setPermError] = useState(null)
  const [analysisError, setAnalysisError] = useState(null)
  const [retryMessage, setRetryMessage] = useState('')
  const [waveformLevels, setWaveformLevels] = useState(IDLE_WAVEFORM_LEVELS)
  const [hasLiveWaveform, setHasLiveWaveform] = useState(false)

  const streamRef = useRef(null)
  const audioContextRef = useRef(null)
  const sourceNodeRef = useRef(null)
  const processorRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const analyserRef = useRef(null)
  const zeroGainRef = useRef(null)
  const waveformFrameRef = useRef(null)
  const analysisAbortControllerRef = useRef(null)
  const analysisTimeoutRef = useRef(null)
  const listeningStartedAtRef = useRef(0)
  const stateRef = useRef('idle')
  const sampleRateRef = useRef(0)
  const rollingChunksRef = useRef([])
  const rollingSampleCountRef = useRef(0)
  const captureChunksRef = useRef([])
  const captureRemainingSamplesRef = useRef(0)
  const finalizeStartedRef = useRef(false)
  const noiseRmsValuesRef = useRef([])
  const backgroundRmsRef = useRef(0.012)
  const activeBurstSecondsRef = useRef(0)

  const setMode = useCallback((nextState) => {
    stateRef.current = nextState
    setCaptureState(nextState)
  }, [])

  const clearAnalysisTimeout = useCallback(() => {
    if (analysisTimeoutRef.current) {
      clearTimeout(analysisTimeoutRef.current)
      analysisTimeoutRef.current = null
    }
  }, [])

  const stopWaveform = useCallback(() => {
    if (waveformFrameRef.current) {
      cancelAnimationFrame(waveformFrameRef.current)
      waveformFrameRef.current = null
    }
    setHasLiveWaveform(false)
  }, [])

  const stopMicrophone = useCallback(() => {
    stopWaveform()

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop()
      } catch {
        // MediaRecorder may already be stopping; cleanup should stay best-effort.
      }
    }
    mediaRecorderRef.current = null

    if (processorRef.current) {
      processorRef.current.onaudioprocess = null
      processorRef.current.disconnect()
      processorRef.current = null
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect()
      sourceNodeRef.current = null
    }

    if (analyserRef.current) {
      analyserRef.current.disconnect()
      analyserRef.current = null
    }

    if (zeroGainRef.current) {
      zeroGainRef.current.disconnect()
      zeroGainRef.current = null
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {})
    }
    audioContextRef.current = null

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }
  }, [stopWaveform])

  const resetCaptureBuffers = useCallback(() => {
    rollingChunksRef.current = []
    rollingSampleCountRef.current = 0
    captureChunksRef.current = []
    captureRemainingSamplesRef.current = 0
    finalizeStartedRef.current = false
    noiseRmsValuesRef.current = []
    backgroundRmsRef.current = 0.012
    activeBurstSecondsRef.current = 0
  }, [])

  const startListening = useCallback(async () => {
    if (analysisAbortControllerRef.current) {
      analysisAbortControllerRef.current.abort()
      analysisAbortControllerRef.current = null
    }
    clearAnalysisTimeout()
    setPermError(null)
    setAnalysisError(null)
    setRetryMessage('')
    setWaveformLevels(IDLE_WAVEFORM_LEVELS)
    resetCaptureBuffers()

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
        },
      })

      const AudioContextCtor = window.AudioContext || window.webkitAudioContext
      if (!AudioContextCtor) throw new Error('AudioContext unavailable')

      const audioContext = new AudioContextCtor()
      await audioContext.resume()

      const sourceNode = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      const processor = audioContext.createScriptProcessor(2048, 1, 1)
      const zeroGain = audioContext.createGain()

      analyser.fftSize = 256
      analyser.smoothingTimeConstant = 0.72
      zeroGain.gain.value = 0

      sourceNode.connect(analyser)
      sourceNode.connect(processor)
      processor.connect(zeroGain)
      zeroGain.connect(audioContext.destination)

      streamRef.current = stream
      audioContextRef.current = audioContext
      sourceNodeRef.current = sourceNode
      analyserRef.current = analyser
      processorRef.current = processor
      zeroGainRef.current = zeroGain
      sampleRateRef.current = audioContext.sampleRate
      listeningStartedAtRef.current = performance.now()

      processor.onaudioprocess = (event) => {
        processAudioChunk(event.inputBuffer.getChannelData(0))
      }

      setMode('listening')
      startWaveform(analyser)
    } catch (err) {
      stopMicrophone()
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        setPermError(copy.recordMicDenied)
      } else if (err.name === 'NotFoundError') {
        setPermError(copy.recordMicNotFound)
      } else {
        setPermError(`${copy.recordMicGeneric}: ${err.message}`)
      }
      setMode('error')
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clearAnalysisTimeout, resetCaptureBuffers, setMode, stopMicrophone])

  const showRetryState = useCallback((message) => {
    clearAnalysisTimeout()
    stopMicrophone()
    resetCaptureBuffers()
    setRetryMessage(message)
    setAnalysisError(message)
    setWaveformLevels(IDLE_WAVEFORM_LEVELS)
    setMode('error')
  }, [clearAnalysisTimeout, resetCaptureBuffers, setMode, stopMicrophone])

  const analyzeCapturedAudio = useCallback(async (blob) => {
    setMode('analyzing')
    setAnalysisError(null)
    setRetryMessage('')

    const controller = new AbortController()
    analysisAbortControllerRef.current = controller
    analysisTimeoutRef.current = setTimeout(() => {
      controller.abort()
    }, PREDICT_TIMEOUT_MS)

    try {
      const formData = new FormData()
      formData.append('file', blob, 'auto-cough-capture.wav')

      const res = await fetch(`${API_BASE_URL}/predict-cough`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      if (!res.ok) {
        throw new Error(`${copy.recordServerStatus} ${res.status}`)
      }

      const data = await res.json()
      const aiResult = {
        label: normalizeBackendLabel(data.label || data.prediction),
        confidence: data.confidence != null
          ? data.confidence
          : data.probability != null
            ? data.probability
            : 0,
        message: data.message || '',
        quality: data.quality || null,
      }

      if (isAcceptedCough(aiResult)) {
        setMode('accepted')
        navigate('/symptoms', { state: { aiResult } })
        return
      }

      showRetryState(getRejectedAnalysisMessage(aiResult))
    } catch (err) {
      const message = err.name === 'AbortError'
        ? PREDICT_TIMEOUT_MESSAGE
        : `${copy.recordAnalysisFail} (${err.message})`
      showRetryState(message)
    } finally {
      clearAnalysisTimeout()
      if (analysisAbortControllerRef.current === controller) {
        analysisAbortControllerRef.current = null
      }
    }
  }, [clearAnalysisTimeout, navigate, setMode, showRetryState])

  const finalizeCapture = useCallback(() => {
    if (finalizeStartedRef.current) return
    finalizeStartedRef.current = true

    const sampleRate = sampleRateRef.current || 44_100
    const capturedSamples = mergeChunks(captureChunksRef.current)
    const wavBlob = pcmToWavBlob(capturedSamples, sampleRate)

    stopMicrophone()
    analyzeCapturedAudio(wavBlob)
  }, [analyzeCapturedAudio, stopMicrophone])

  const pushRollingChunk = useCallback((chunk, sampleRate) => {
    rollingChunksRef.current.push(chunk)
    rollingSampleCountRef.current += chunk.length

    const maxRollingSamples = Math.ceil(PRE_ROLL_SECONDS * sampleRate)
    while (rollingSampleCountRef.current > maxRollingSamples && rollingChunksRef.current.length > 1) {
      const removed = rollingChunksRef.current.shift()
      rollingSampleCountRef.current -= removed.length
    }
  }, [])

  const startWaveform = useCallback((analyser) => {
    const dataArray = new Uint8Array(analyser.frequencyBinCount)
    setHasLiveWaveform(true)

    const draw = () => {
      analyser.getByteTimeDomainData(dataArray)
      const chunkSize = Math.max(1, Math.floor(dataArray.length / WAVEFORM_BAR_COUNT))
      const nextLevels = Array.from({ length: WAVEFORM_BAR_COUNT }, (_, barIndex) => {
        const start = barIndex * chunkSize
        const end = Math.min(dataArray.length, start + chunkSize)
        let total = 0

        for (let index = start; index < end; index += 1) {
          total += Math.abs(dataArray[index] - 128) / 128
        }

        const average = total / Math.max(1, end - start)
        return Math.min(1, Math.max(0.12, average * 4.4))
      })

      setWaveformLevels(nextLevels)
      waveformFrameRef.current = requestAnimationFrame(draw)
    }

    draw()
  }, [])

  const processAudioChunk = useCallback((inputSamples) => {
    const sampleRate = sampleRateRef.current
    if (!sampleRate || stateRef.current === 'analyzing' || stateRef.current === 'accepted') return

    const chunk = new Float32Array(inputSamples)
    const features = getAudioFeatures(chunk)
    const elapsedSeconds = (performance.now() - listeningStartedAtRef.current) / 1000

    pushRollingChunk(chunk, sampleRate)

    if (elapsedSeconds <= NOISE_CALIBRATION_SECONDS) {
      noiseRmsValuesRef.current.push(features.rms)
      const values = noiseRmsValuesRef.current
      backgroundRmsRef.current = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length)
      return
    }

    if (stateRef.current === 'capturing') {
      captureChunksRef.current.push(chunk)
      captureRemainingSamplesRef.current -= chunk.length
      if (captureRemainingSamplesRef.current <= 0) finalizeCapture()
      return
    }

    if (stateRef.current !== 'listening' && stateRef.current !== 'cough_candidate_detected') return

    const threshold = Math.max(MIN_TRIGGER_RMS, backgroundRmsRef.current * TRIGGER_MULTIPLIER)
    const chunkSeconds = chunk.length / sampleRate
    const isActive = features.rms >= threshold && features.peak >= MIN_TRIGGER_PEAK

    if (isActive) {
      activeBurstSecondsRef.current += chunkSeconds
      if (activeBurstSecondsRef.current >= MIN_BURST_SECONDS && activeBurstSecondsRef.current <= MAX_BURST_SECONDS) {
        setMode('cough_candidate_detected')
        captureChunksRef.current = rollingChunksRef.current.map((rollingChunk) => new Float32Array(rollingChunk))
        captureRemainingSamplesRef.current = Math.ceil(POST_ROLL_SECONDS * sampleRate)
        setMode('capturing')
      }
      return
    }

    if (activeBurstSecondsRef.current > MAX_BURST_SECONDS) {
      activeBurstSecondsRef.current = 0
    } else {
      activeBurstSecondsRef.current = Math.max(0, activeBurstSecondsRef.current - chunkSeconds * 0.5)
    }
  }, [finalizeCapture, pushRollingChunk, setMode])

  const cancelListening = useCallback(() => {
    if (analysisAbortControllerRef.current) {
      analysisAbortControllerRef.current.abort()
      analysisAbortControllerRef.current = null
    }
    clearAnalysisTimeout()
    stopMicrophone()
    resetCaptureBuffers()
    setMode('idle')
    setAnalysisError(null)
    setRetryMessage('')
    setWaveformLevels(IDLE_WAVEFORM_LEVELS)
  }, [clearAnalysisTimeout, resetCaptureBuffers, setMode, stopMicrophone])

  useEffect(() => {
    return () => {
      if (analysisAbortControllerRef.current) {
        analysisAbortControllerRef.current.abort()
        analysisAbortControllerRef.current = null
      }
      clearAnalysisTimeout()
      stopMicrophone()
    }
  }, [clearAnalysisTimeout, stopMicrophone])

  const isMicActive = captureState === 'listening' || captureState === 'cough_candidate_detected' || captureState === 'capturing'
  const isBusy = isMicActive || captureState === 'analyzing' || captureState === 'retry_listening' || captureState === 'accepted'
  const primaryButtonText = captureState === 'error'
    ? 'ลองตรวจจับใหม่'
    : isMicActive
      ? 'หยุดตรวจจับ'
      : 'เริ่มตรวจจับ'
  const primaryButtonAria = captureState === 'error'
    ? 'ลองตรวจจับเสียงไอใหม่'
    : isMicActive
      ? 'หยุดตรวจจับเสียงไอ'
      : 'เริ่มตรวจจับเสียงไอ'

  return (
    <div className="record-page">
      <button className="back-link" onClick={() => navigate('/')} type="button">
        {copy.commonBack}
      </button>

      <div className="step-indicator mvp-step-indicator" aria-label={copy.recordMvpStep}>
        <span className="step-dot active"></span>
        <span className="step-line"></span>
        <span className="step-dot"></span>
        <span className="step-line"></span>
        <span className="step-dot"></span>
        <span>{copy.recordMvpStep}</span>
      </div>

      <section className="page-heading record-heading">
        <div>
          <p className="section-kicker">{copy.recordMvpKicker}</p>
          <h2>ตรวจจับเสียงไอด้วย AI</h2>
          <p>กดเริ่มตรวจจับ แล้วไอ 1–2 ครั้ง ระบบจะรับเฉพาะช่วงเสียงที่ AI มั่นใจว่าเป็นเสียงไอ 70% ขึ้นไป</p>
        </div>
      </section>

      {permError && (
        <div className="error-message mt-16">
          <strong>{copy.recordMicFailTitle}</strong>
          <span>{permError}</span>
        </div>
      )}

      <section className={`record-panel auto-capture-panel state-${captureState}`}>
        <div className="record-core-banner">
          <span aria-hidden="true">AI</span>
          <div>
            <strong>AI Cough Detection 70%+</strong>
            <p>ไมค์เปิดเฉพาะตอนกดเริ่มตรวจจับ และจะไปขั้นตอนถัดไปเมื่อเสียงไอชัดพอเท่านั้น</p>
          </div>
        </div>

        <div
          className={`waveform ${isMicActive ? 'active' : ''} ${hasLiveWaveform ? 'live' : 'fallback'}`}
          aria-hidden="true"
        >
          {waveformLevels.map((level, index) => (
            <span key={index} style={{ '--bar-index': index, '--bar-level': level }} />
          ))}
        </div>

        <div className="record-control">
          <button
            className={`record-btn ${isMicActive ? 'recording' : ''}`}
            onClick={isMicActive || captureState === 'retry_listening' ? cancelListening : startListening}
            disabled={captureState === 'analyzing' || captureState === 'accepted'}
            aria-label={primaryButtonAria}
            type="button"
          >
            <span className="record-btn-icon">
              {primaryButtonText}
            </span>
          </button>
          <span className="listening-ring" aria-hidden="true" />
        </div>

        <div className="record-status">
          <span>{retryMessage || STATE_TEXT[captureState]}</span>
        </div>

        {isBusy && (
          <div className="privacy-note">
            <span aria-hidden="true" />
            <p>กำลังตรวจจับเสียงไอ ไมค์จะไม่ทำงานเบื้องหลัง</p>
          </div>
        )}
      </section>

      {captureState === 'analyzing' && (
        <div className="loading-overlay">
          <img className="loading-logo" src={logos.abstractMark} alt={copy.recordLoadingAlt} />
          <span className="loading-text">{STATE_TEXT.analyzing}</span>
        </div>
      )}

      {analysisError && (
        <div className="error-message mt-16">
          <strong>{copy.recordAnalysisFailTitle}</strong>
          <span>{analysisError}</span>
          <button className="record-retry-btn" onClick={startListening} type="button">
            ลองตรวจจับใหม่
          </button>
        </div>
      )}
    </div>
  )
}

export default RecordPage
