import { useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { copy } from '../assets/copy'
import { logos } from '../assets/logos'
import './ResultPage.css'

const SYMPTOM_LABELS = {
  cough: copy.symptomCough,
  sore_throat: copy.symptomSoreThroat,
  itchy_eyes: copy.symptomItchyEyes,
  nasal_congestion: copy.symptomNasal,
  breathing_discomfort: copy.symptomBreathing,
  fatigue: copy.symptomFatigue,
}

const OUTDOOR_LABELS = {
  less_1: copy.outdoorLess,
  '1_3': copy.outdoorMedium,
  more_3: copy.outdoorMore,
}

const MASK_LABELS = {
  always: copy.maskAlways,
  sometimes: copy.maskSometimes,
  never: copy.maskNever,
}

const getAiLabel = (label) => {
  if (label === 'cough') return copy.resultDetectedCough
  if (label === 'uncertain_cough') return copy.recordLabelUncertainCough
  if (label === 'non_cough') return copy.recordLabelNonCough
  if (label === 'too_quiet') return copy.recordLabelTooQuiet
  if (label === 'too_short') return copy.recordLabelTooShort
  if (label === 'noise' || label === 'unclear') return copy.recordLabelUnclear
  return copy.recordLabelUnknown
}

const getPm25Label = (value) => {
  if (value < 37.5) return copy.resultPm25Low
  if (value <= 75) return copy.resultPm25Medium
  return copy.resultPm25High
}

const PM25_UNIT_SHORT = '\u00b5g/m\u00b3'

const resultCopy = {
  noiseRecommendation: copy.resultNoiseRecommendation,
  noiseIgnored: copy.resultNoiseIgnored,
  pm25Status: copy.resultPm25Status,
  scoreTotal: copy.resultScoreTotal,
  disclaimer: copy.resultDisclaimerClear,
}

const RESULT_SUMMARIES = {
  low: {
    description: copy.resultSummaryLow,
    actions: [copy.adviceLow1Clear, copy.adviceLow2Clear, copy.adviceLow3Clear],
  },
  medium: {
    description: copy.resultSummaryMedium,
    actions: [
      copy.adviceMedium1Clear,
      copy.adviceMedium2Clear,
      copy.adviceMedium3Clear,
      copy.adviceMedium4Clear,
    ],
  },
  high: {
    description: copy.resultSummaryHigh,
    actions: [
      copy.adviceHigh1Clear,
      copy.adviceHigh2Clear,
      copy.adviceHigh3Clear,
      copy.adviceHigh4Clear,
    ],
  },
}

const isUnclearAudio = (label) => !label || label === 'noise' || label === 'unknown' || label === 'unclear'
const isRetryAudio = (label) => isUnclearAudio(label) || label === 'too_quiet' || label === 'too_short'
const isPartialAudio = (label) => label === 'uncertain_cough'

function ResultPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { aiResult, pm25, symptoms = [], outdoor, mask } = location.state || {}

  const riskData = useMemo(() => {
    let score = 0
    const breakdown = []
    const pm25Val = Number(pm25) || 0

    if (pm25Val < 37.5) {
      breakdown.push({ label: copy.resultBreakdownPm25, detail: `${pm25Val} ${copy.resultPm25Unit}`, points: 0 })
    } else if (pm25Val <= 75) {
      score += 1
      breakdown.push({ label: copy.resultBreakdownPm25, detail: `${pm25Val} ${copy.resultPm25Unit}`, points: 1 })
    } else {
      score += 2
      breakdown.push({ label: copy.resultBreakdownPm25, detail: `${pm25Val} ${copy.resultPm25Unit}`, points: 2 })
    }

    if (aiResult?.label === 'cough') {
      score += 2
      breakdown.push({ label: copy.resultBreakdownAudio, detail: copy.recordLabelCough, points: 2 })
    } else if (aiResult?.label === 'uncertain_cough') {
      score += 1
      breakdown.push({
        label: copy.resultBreakdownAudio,
        detail: copy.recordLabelUncertainCough,
        points: 1,
      })
    } else if (isRetryAudio(aiResult?.label)) {
      breakdown.push({
        label: copy.resultBreakdownAudio,
        detail: aiResult?.message || resultCopy.noiseIgnored,
        points: 0,
        tone: 'muted',
      })
    } else {
      breakdown.push({ label: copy.resultBreakdownAudio, detail: getAiLabel(aiResult?.label), points: 0 })
    }

    const symptomCount = symptoms.length
    if (symptomCount >= 3) {
      score += 2
      breakdown.push({
        label: copy.resultBreakdownSymptoms,
        detail: `${copy.resultSymptomFound} ${symptomCount} ${copy.resultItems}`,
        points: 2,
      })
    } else if (symptomCount >= 1) {
      score += 1
      breakdown.push({
        label: copy.resultBreakdownSymptoms,
        detail: `${copy.resultSymptomFound} ${symptomCount} ${copy.resultItems}`,
        points: 1,
      })
    } else {
      breakdown.push({ label: copy.resultBreakdownSymptoms, detail: copy.resultNoJointSymptoms, points: 0 })
    }

    if (outdoor === 'more_3' || mask === 'never') {
      score += 1
      const reasons = []
      if (outdoor === 'more_3') reasons.push(copy.resultReasonOutdoor)
      if (mask === 'never') reasons.push(copy.resultReasonNoMask)
      breakdown.push({ label: copy.resultBreakdownExposure, detail: reasons.join(', '), points: 1 })
    } else {
      breakdown.push({ label: copy.resultBreakdownExposure, detail: copy.resultGeneralWatch, points: 0 })
    }

    let level = 'low'
    let levelLabel = copy.resultRiskLow
    let levelDescription = copy.resultRiskLowDesc

    if (score >= 6) {
      level = 'high'
      levelLabel = copy.resultRiskHigh
      levelDescription = copy.resultRiskHighDesc
    } else if (score >= 3) {
      level = 'medium'
      levelLabel = copy.resultRiskMedium
      levelDescription = copy.resultRiskMediumDesc
    }

    return { score, breakdown, level, levelLabel, levelDescription, maxScore: 7 }
  }, [aiResult, pm25, symptoms, outdoor, mask])

  const advice = useMemo(() => {
    return RESULT_SUMMARIES[riskData.level]?.actions || RESULT_SUMMARIES.low.actions
  }, [riskData.level])

  const resultSummary = RESULT_SUMMARIES[riskData.level] || RESULT_SUMMARIES.low
  const audioNeedsRetry = isRetryAudio(aiResult?.label)
  const audioPartial = isPartialAudio(aiResult?.label)
  const audioUsable = !audioNeedsRetry
  const aiAnalysis = useMemo(() => {
    if (aiResult?.label === 'cough') {
      return {
        tone: 'cough',
        label: copy.resultDetectedCough,
        explanation: copy.resultAiExplainCough,
        usable: copy.resultAiUsableYes,
      }
    }

    if (aiResult?.label === 'uncertain_cough') {
      return {
        tone: 'uncertain',
        label: copy.recordLabelUncertainCough,
        explanation: aiResult?.message || copy.resultAiExplainUncertainCough,
        usable: copy.resultAiUsablePartial,
      }
    }

    if (aiResult?.label === 'non_cough') {
      return {
        tone: 'non-cough',
        label: copy.recordLabelNonCough,
        explanation: copy.resultAiExplainNonCough,
        usable: copy.resultAiUsableYes,
      }
    }

    if (aiResult?.label === 'too_quiet') {
      return {
        tone: 'unclear',
        label: copy.recordLabelTooQuiet,
        explanation: aiResult?.message || copy.resultAiExplainTooQuiet,
        usable: copy.resultAiUsableRetry,
      }
    }

    if (aiResult?.label === 'too_short') {
      return {
        tone: 'unclear',
        label: copy.recordLabelTooShort,
        explanation: aiResult?.message || copy.resultAiExplainTooShort,
        usable: copy.resultAiUsableRetry,
      }
    }

    return {
      tone: 'unclear',
      label: copy.recordLabelUnclear,
      explanation: copy.resultAiExplainUnclear,
      usable: copy.resultAiUsableRetry,
    }
  }, [aiResult])

  if (!location.state) {
    return (
      <div className="result-page empty-result">
        <section className="card text-center">
          <h2>{copy.resultNoDataTitle}</h2>
          <p className="text-secondary mt-8">{copy.resultNoDataDesc}</p>
          <button className="btn btn-primary btn-lg mt-32" onClick={() => navigate('/')} type="button">
            {copy.resultStart}
          </button>
        </section>
      </div>
    )
  }

  return (
    <div className="result-page">
      <div className="step-indicator mvp-step-indicator" aria-label={copy.resultMvpStep}>
        <span className="step-dot done"></span>
        <span className="step-line done"></span>
        <span className="step-dot done"></span>
        <span className="step-line done"></span>
        <span className="step-dot active"></span>
        <span>{copy.resultMvpStep}</span>
      </div>

      <section className="page-heading result-heading">
        <div>
          <p className="section-kicker">{copy.resultMvpKicker}</p>
          <h2>{copy.resultMvpTitle}</h2>
          <p>{copy.resultMvpDescription}</p>
        </div>
        <img className="page-symbol result-emblem" src={logos.emblem} alt={copy.resultEmblemAlt} />
      </section>

      <section className={`ai-analysis-card ai-analysis-${aiAnalysis.tone}`}>
        <div className="ai-analysis-header">
          <img src={logos.abstractMark} alt="" aria-hidden="true" />
          <div>
            <p className="section-kicker">{copy.resultPrimaryAudio}</p>
            <h3>{copy.resultAiCardTitle}</h3>
          </div>
        </div>
        <div className="ai-analysis-grid">
          <div className="ai-analysis-main">
            <span>{copy.resultAiLabel}</span>
            <strong>{aiAnalysis.label}</strong>
          </div>
          <div>
            <span>{copy.resultAiConfidence}</span>
            <strong>{Math.round((aiResult?.confidence || 0) * 100)}%</strong>
          </div>
          <div className={audioUsable ? 'usable-yes' : 'usable-no'}>
            <span>{copy.resultAiUsable}</span>
            <strong>{aiAnalysis.usable}</strong>
          </div>
        </div>
        <p>{aiAnalysis.explanation}</p>
      </section>

      <section className="summary-card card supporting-context-card">
        <div className="summary-header">
          <h3>{copy.resultSupportingEnvSymptoms || copy.resultDataUsedAiFirst}</h3>
        </div>
        <div className="summary-list">
          <div className="summary-row">
            <span>{copy.resultPm25}</span>
            <div className="summary-value pm25-value">
              <strong>{copy.resultPm25}: {pm25} {PM25_UNIT_SHORT}</strong>
              <b>{resultCopy.pm25Status}: {getPm25Label(Number(pm25))}</b>
            </div>
          </div>
          <div className="summary-row">
            <span>{copy.resultSymptoms}</span>
            <strong>
              {symptoms.length > 0
                ? symptoms.map((symptom) => SYMPTOM_LABELS[symptom] || symptom).join(', ')
                : copy.resultNoSymptoms}
            </strong>
          </div>
          <div className="summary-row">
            <span>{copy.resultOutdoor}</span>
            <strong>{OUTDOOR_LABELS[outdoor] || '-'}</strong>
          </div>
          <div className="summary-row">
            <span>{copy.resultMask}</span>
            <strong>{MASK_LABELS[mask] || '-'}</strong>
          </div>
        </div>
      </section>

      <section className={`result-summary-card result-summary-${riskData.level}`}>
        <p className="section-kicker">{audioNeedsRetry ? copy.resultSupportingOnly : copy.resultSummaryKicker}</p>
        {audioNeedsRetry ? (
          <h3>{aiResult?.label === 'too_quiet' || aiResult?.label === 'too_short' ? copy.resultIncompleteQualityTitle : copy.resultUnableAnalyzeAudio}</h3>
        ) : (
          <h3>
            {copy.resultSummaryTitlePrefix} <span className={`risk-${riskData.level}`}>{riskData.levelLabel}</span>
          </h3>
        )}
        <p>{audioNeedsRetry ? copy.resultRiskContextUnclear : resultSummary.description}</p>
        {!audioNeedsRetry && (
          <div className="next-action-panel">
            <strong>{copy.resultNowActionTitle}</strong>
            <div className="next-action-grid">
              {resultSummary.actions.slice(0, 2).map((item, index) => (
                <article key={item} className="next-action-card">
                  <span>{index + 1}</span>
                  <p>{item}</p>
                </article>
              ))}
            </div>
          </div>
        )}
      </section>

      <section
        className={`risk-overview risk-${riskData.level}-bg`}
        style={{ '--score-progress': `${(riskData.score / riskData.maxScore) * 360}deg` }}
      >
        <div>
          <p className="risk-label">{audioNeedsRetry ? copy.resultSupportingOnly : copy.resultPrelimRiskTitle}</p>
          <h3 className={`risk-${riskData.level}`}>{riskData.levelLabel}</h3>
          <p>{riskData.levelDescription}</p>
          <small className="score-context">
            {audioNeedsRetry ? copy.resultRiskContextUnclear : copy.resultRiskContextAi}
            {audioPartial ? ` ${copy.resultAiUsablePartial}` : ''}
          </small>
        </div>
        <div className={`risk-score-display risk-score-${riskData.level}`}>
          <small>{resultCopy.scoreTotal}</small>
          <strong>{riskData.score}</strong>
          <span>{riskData.score} {copy.resultFrom} {riskData.maxScore}</span>
        </div>
      </section>

      <section className={`advice-card card advice-${riskData.level}`}>
        <div className="summary-header">
          <h3>{copy.resultAdvice}</h3>
        </div>
        <div className="advice-action-grid">
          {advice.map((item, index) => (
            <article key={item} className="advice-action-card">
              <span>{index + 1}</span>
              <p>{item}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="breakdown-card card">
        <div className="summary-header">
          <h3>{copy.resultBreakdown}</h3>
        </div>
        <div className="breakdown-list">
          <div className="breakdown-group-label">{copy.resultPrimaryAudio}</div>
          {riskData.breakdown.slice(1, 2).map((item) => (
            <div key={`${item.label}-${item.detail}`} className={`breakdown-row ${item.tone === 'muted' ? 'breakdown-row-muted' : ''}`}>
              <div>
                <strong>{item.label}</strong>
                <span>{item.detail}</span>
              </div>
              <b className={item.points > 0 ? 'has-points' : ''}>+{item.points}</b>
            </div>
          ))}
          <div className="breakdown-group-label">{copy.resultSupportingData}</div>
          {[riskData.breakdown[0], ...riskData.breakdown.slice(2)].map((item) => (
            <div key={`${item.label}-${item.detail}`} className={`breakdown-row ${item.tone === 'muted' ? 'breakdown-row-muted' : ''}`}>
              <div>
                <strong>{item.label}</strong>
                <span>{item.detail}</span>
              </div>
              <b className={item.points > 0 ? 'has-points' : ''}>+{item.points}</b>
            </div>
          ))}
          <div className="breakdown-total">
            <span>{copy.resultTotal}</span>
            <strong className={`risk-${riskData.level}`}>{riskData.score} {copy.resultFrom} {riskData.maxScore}</strong>
          </div>
        </div>
      </section>

      <section className="notice-panel">
        <h3>{copy.resultHealthWarning}</h3>
        <p>{copy.resultWarningText}</p>
      </section>

      <section className="disclaimer-panel">
        <p>{copy.resultDisclaimerSimple || resultCopy.disclaimer}</p>
      </section>

      <div className="result-actions">
        <button
          className="btn btn-primary btn-lg btn-full"
          onClick={() => navigate(audioNeedsRetry ? '/record' : '/')}
          type="button"
        >
          {audioNeedsRetry ? copy.resultReRecordCoughCta : copy.resultRestart}
        </button>
      </div>
    </div>
  )
}

export default ResultPage
