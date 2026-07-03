import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { copy } from '../assets/copy'
import './SymptomFormPage.css'

const SYMPTOMS = [
  { id: 'cough', label: copy.symptomCough },
  { id: 'sore_throat', label: copy.symptomSoreThroat },
  { id: 'itchy_eyes', label: copy.symptomItchyEyes },
  { id: 'nasal_congestion', label: copy.symptomNasal },
  { id: 'breathing_discomfort', label: copy.symptomBreathing },
  { id: 'fatigue', label: copy.symptomFatigue },
]

const OUTDOOR_OPTIONS = [
  { value: 'less_1', label: copy.outdoorLess },
  { value: '1_3', label: copy.outdoorMedium },
  { value: 'more_3', label: copy.outdoorMore },
]

const MASK_OPTIONS = [
  { value: 'always', label: copy.maskAlways },
  { value: 'sometimes', label: copy.maskSometimes },
  { value: 'never', label: copy.maskNever },
]

const getAiLabel = (label) => {
  if (label === 'cough') return copy.recordLabelCough
  if (label === 'uncertain_cough') return copy.recordLabelUncertainCough
  if (label === 'non_cough') return copy.recordLabelNonCough
  if (label === 'too_quiet') return copy.recordLabelTooQuiet
  if (label === 'too_short') return copy.recordLabelTooShort
  if (label === 'noise' || label === 'unclear') return copy.recordLabelUnclear
  return copy.recordLabelUnknown
}

function SymptomFormPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const aiResult = location.state?.aiResult || null

  const [pm25, setPm25] = useState('')
  const [selectedSymptoms, setSelectedSymptoms] = useState([])
  const [outdoor, setOutdoor] = useState('')
  const [mask, setMask] = useState('')
  const [errors, setErrors] = useState({})

  const toggleSymptom = (id) => {
    setSelectedSymptoms((prev) =>
      prev.includes(id) ? prev.filter((symptom) => symptom !== id) : [...prev, id]
    )
  }

  const validate = () => {
    const newErrors = {}

    if (!pm25 || Number.isNaN(Number(pm25)) || Number(pm25) < 0) {
      newErrors.pm25 = copy.symptomPm25Error
    }

    if (!outdoor) {
      newErrors.outdoor = copy.symptomOutdoorError
    }

    if (!mask) {
      newErrors.mask = copy.symptomMaskError
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = (event) => {
    event.preventDefault()
    if (!validate()) return

    navigate('/result', {
      state: {
        aiResult,
        pm25: Number(pm25),
        symptoms: selectedSymptoms,
        outdoor,
        mask,
      },
    })
  }

  return (
    <div className="symptom-page">
      <button className="back-link" onClick={() => navigate('/record')} type="button">
        {copy.commonBack}
      </button>

      <div className="step-indicator mvp-step-indicator" aria-label="ขั้นตอนที่ 2 จาก 3">
        <span className="step-dot done"></span>
        <span className="step-line done"></span>
        <span className="step-dot active"></span>
        <span className="step-line"></span>
        <span className="step-dot"></span>
        <span>ขั้นตอนที่ 2 จาก 3</span>
      </div>

      <section className="page-heading">
        <p className="section-kicker">{copy.symptomSupportingKicker}</p>
        <h2>{copy.symptomSupportingTitle}</h2>
        <p>{copy.symptomSupportingDescription}</p>
      </section>

      {aiResult && (
        <section className="ai-summary card">
          <span className="ai-summary-badge">{copy.symptomAiBadge}</span>
          <div>
            <strong>{getAiLabel(aiResult.label)}</strong>
            <p>{copy.commonConfidence} {Math.round(aiResult.confidence * 100)}%</p>
          </div>
        </section>
      )}

      <form className="assessment-form" onSubmit={handleSubmit} noValidate>
        <div className="form-flow-note">
          <span>{copy.formFlowSupportingLabel}</span>
          <strong>{copy.formFlowSupportingStrong}</strong>
        </div>

        <section className="form-section">
          <label className="form-label" htmlFor="pm25">
            {copy.symptomPm25Label} <span className="required-star">*</span>
          </label>
          <input
            id="pm25"
            data-testid="pm25-input"
            type="number"
            className={`form-input ${errors.pm25 ? 'input-error' : ''}`}
            placeholder={copy.symptomPm25Placeholder}
            value={pm25}
            onChange={(event) => setPm25(event.target.value)}
            min="0"
            step="0.1"
          />
          {errors.pm25 && <span className="field-error">{errors.pm25}</span>}
          <p className="field-hint">{copy.symptomPm25Hint}</p>
        </section>

        <section className="form-section">
          <fieldset>
            <legend className="form-label">{copy.symptomCurrentLegend}</legend>
            <div className="symptom-grid">
              {SYMPTOMS.map((symptom) => (
                <label
                  key={symptom.id}
                  data-testid={`symptom-${symptom.id}`}
                  className={`checkbox-card ${selectedSymptoms.includes(symptom.id) ? 'active' : ''}`}
                >
                  <input
                    type="checkbox"
                    checked={selectedSymptoms.includes(symptom.id)}
                    onChange={() => toggleSymptom(symptom.id)}
                  />
                  <span className="check-icon" aria-hidden="true" />
                  <span className="symptom-label-text">{symptom.label}</span>
                </label>
              ))}
            </div>
          </fieldset>
        </section>

        <section className="form-section">
          <fieldset>
            <legend className="form-label">
              {copy.symptomOutdoorLegend} <span className="required-star">*</span>
            </legend>
            <div className="radio-group">
              {OUTDOOR_OPTIONS.map((option) => (
                <label
                  key={option.value}
                  data-testid={`outdoor-${option.value}`}
                  className={`radio-card ${outdoor === option.value ? 'active' : ''}`}
                >
                  <input
                    type="radio"
                    name="outdoor"
                    value={option.value}
                    checked={outdoor === option.value}
                    onChange={(event) => setOutdoor(event.target.value)}
                  />
                  <span className="radio-icon" aria-hidden="true" />
                  <span>{option.label}</span>
                </label>
              ))}
            </div>
            {errors.outdoor && <span className="field-error">{errors.outdoor}</span>}
          </fieldset>
        </section>

        <section className="form-section">
          <fieldset>
            <legend className="form-label">
              {copy.symptomMaskLegend} <span className="required-star">*</span>
            </legend>
            <div className="radio-group">
              {MASK_OPTIONS.map((option) => (
                <label
                  key={option.value}
                  data-testid={`mask-${option.value}`}
                  className={`radio-card ${mask === option.value ? 'active' : ''}`}
                >
                  <input
                    type="radio"
                    name="mask"
                    value={option.value}
                    checked={mask === option.value}
                    onChange={(event) => setMask(event.target.value)}
                  />
                  <span className="radio-icon" aria-hidden="true" />
                  <span>{option.label}</span>
                </label>
              ))}
            </div>
            {errors.mask && <span className="field-error">{errors.mask}</span>}
          </fieldset>
        </section>

        <button type="submit" className="btn btn-primary btn-lg btn-full" data-testid="submit-assessment">
          {copy.symptomSubmit}
        </button>
      </form>
    </div>
  )
}

export default SymptomFormPage
