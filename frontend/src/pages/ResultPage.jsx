import { useMemo } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { copy } from '../assets/copy'
import { logos } from '../assets/logos'
import './ResultPage.css'

const PM25_UNIT_SHORT = '\u00b5g/m\u00b3'
const HIGH_CONFIDENCE_THRESHOLD = 0.75

const AUDIO_LABEL_TEXT = {
  dry_cough_like: 'ลักษณะคล้ายไอแห้ง',
  wet_cough_like: 'ลักษณะคล้ายไอมีเสมหะ',
  frequent_cough_like: 'ลักษณะคล้ายไอถี่หรือต่อเนื่อง',
  normal_cough_like: 'ลักษณะคล้ายเสียงไอทั่วไป',
  non_cough: 'ไม่พบเสียงไอชัดเจน',
  noise: 'เสียงรบกวน',
  unclear: 'เสียงไม่ชัดเจน',
}

const AUDIO_QUALITY_LABELS = {
  good: 'ดี',
  too_quiet: 'เสียงเบาเกินไป',
  too_short: 'เสียงสั้นเกินไป',
  noise: 'มีเสียงรบกวน',
  unclear: 'ไม่ชัดเจน',
}

const COUGH_AUDIO_LABELS = new Set([
  'dry_cough_like',
  'wet_cough_like',
  'frequent_cough_like',
  'normal_cough_like',
])

const COUGH_PATTERN_SCORES = {
  normal_cough_like: 1,
  dry_cough_like: 1,
  wet_cough_like: 2,
  frequent_cough_like: 2,
  non_cough: 0,
  noise: 0,
  unclear: 0,
}

const SYMPTOM_LABELS = {
  cough: copy.symptomCough,
  sore_throat: copy.symptomSoreThroat,
  itchy_eyes: copy.symptomItchyEyes,
  nasal_congestion: copy.symptomNasal,
  breathing_discomfort: copy.symptomBreathing,
  fatigue: copy.symptomFatigue,
  severe_breathing_difficulty: 'หายใจลำบากรุนแรง',
  chest_pain: 'เจ็บหน้าอก',
  coughing_blood: 'ไอมีเลือดปน',
  severe_symptoms: 'อาการรุนแรงหรือแย่ลงเร็ว',
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

const RED_FLAG_SYMPTOMS = new Set([
  'severe_breathing_difficulty',
  'chest_pain',
  'coughing_blood',
  'severe_symptoms',
])

const RISK_LEVELS = {
  low: {
    label: 'ต่ำ',
    title: 'ความเสี่ยงเบื้องต้นต่ำ',
    description: 'คะแนนรวมอยู่ในช่วงต่ำ ควรติดตามค่า PM2.5 และสังเกตอาการต่อเนื่อง',
    recommendations: [
      'ติดตามค่า PM2.5 และหลีกเลี่ยงฝุ่นเมื่อค่าเริ่มสูง',
      'ดื่มน้ำและพักเสียง หากมีอาการระคายคอ',
      'สังเกตอาการต่อเนื่อง หากอาการเพิ่มขึ้นควรแจ้งผู้ปกครอง',
    ],
  },
  medium: {
    label: 'ปานกลาง',
    title: 'ความเสี่ยงเบื้องต้นปานกลาง',
    description: 'คะแนนรวมอยู่ในช่วงที่ควรลดการสัมผัสฝุ่นและติดตามอาการใกล้ชิดขึ้น',
    recommendations: [
      'ลดกิจกรรมนอกอาคารและอยู่ในพื้นที่อากาศถ่ายเทดี',
      'สวมหน้ากากที่เหมาะสมเมื่อต้องออกนอกอาคาร',
      'ติดตามอาการและค่า PM2.5 ในช่วง 24 ชั่วโมงถัดไป',
      'แจ้งผู้ปกครองหากหายใจไม่สบายหรืออาการแย่ลง',
    ],
  },
  high: {
    label: 'สูง',
    title: 'ความเสี่ยงเบื้องต้นสูง',
    description: 'คะแนนรวมอยู่ในช่วงสูง ควรลดการสัมผัสฝุ่นและให้ผู้ปกครองช่วยติดตามอาการ',
    recommendations: [
      'หลีกเลี่ยงกิจกรรมนอกอาคารและลดการสัมผัสฝุ่น',
      'สวมหน้ากากที่เหมาะสมเมื่อต้องอยู่ในพื้นที่เสี่ยง',
      'พักผ่อน ดื่มน้ำ และแจ้งผู้ปกครองให้ช่วยติดตามอาการ',
      'หากมีอาการรุนแรง ควรปรึกษาแพทย์/บุคลากรทางการแพทย์',
    ],
  },
}

const SAFETY_NOTICE =
  'ระบบนี้เป็นเพียงเครื่องมือคัดกรองและสร้างความตระหนักด้านสุขภาพเบื้องต้น ไม่ใช่เครื่องมือวินิจฉัยโรค'

const RED_FLAG_WARNING =
  'พบอาการที่ควรเฝ้าระวังเป็นพิเศษ ควรแจ้งผู้ปกครองหรือปรึกษาแพทย์/บุคลากรทางการแพทย์'

const formatPercent = (value) => `${Math.round((Number(value) || 0) * 100)}%`

const getPm25Label = (value) => {
  if (value < 37.5) return 'อยู่ในเกณฑ์ต่ำ'
  if (value <= 75) return 'อยู่ในเกณฑ์ปานกลาง'
  return 'อยู่ในเกณฑ์สูง'
}

const getAudioLabel = (aiResult) => {
  if (aiResult?.audioLabel) return aiResult.audioLabel
  if (aiResult?.audio_label) return aiResult.audio_label
  if (aiResult?.label === 'cough' || aiResult?.label === 'uncertain_cough') return 'normal_cough_like'
  if (aiResult?.label === 'non_cough') return 'non_cough'
  if (aiResult?.label === 'noise') return 'noise'
  return 'unclear'
}

const getCoughTypeText = (aiResult) => {
  const audioLabel = getAudioLabel(aiResult)
  return aiResult?.coughTypeText || aiResult?.cough_type_text || AUDIO_LABEL_TEXT[audioLabel] || AUDIO_LABEL_TEXT.unclear
}

const getAudioQuality = (aiResult) => {
  return aiResult?.audioQuality || aiResult?.audio_quality || aiResult?.quality?.label || 'good'
}

const getFeatureValue = (features, key, fallback = 0) => {
  const value = Number(features?.[key])
  return Number.isFinite(value) ? value : fallback
}

const getSymptomNames = (symptoms) => {
  return symptoms.map((symptom) => SYMPTOM_LABELS[symptom] || symptom)
}

function ResultPage() {
  const navigate = useNavigate()
  const location = useLocation()
  const { aiResult, pm25, symptoms = [], outdoor, mask } = location.state || {}

  const audioLabel = getAudioLabel(aiResult)
  const coughTypeText = getCoughTypeText(aiResult)
  const confidence = Number(aiResult?.confidence) || 0
  const audioQuality = getAudioQuality(aiResult)
  const coughFeatures = aiResult?.coughFeatures || aiResult?.cough_features || {}
  const symptomNames = getSymptomNames(symptoms)
  const hasRedFlag = symptoms.some((symptom) => RED_FLAG_SYMPTOMS.has(symptom))

  const riskData = useMemo(() => {
    const pm25Val = Number(pm25) || 0
    const coughDetected = Boolean(
      aiResult?.coughDetected ||
      aiResult?.cough_detected ||
      COUGH_AUDIO_LABELS.has(audioLabel)
    )

    const audioScore = coughDetected && COUGH_AUDIO_LABELS.has(audioLabel)
      ? confidence >= HIGH_CONFIDENCE_THRESHOLD
        ? 2
        : 1
      : 0

    const coughPatternScore = COUGH_PATTERN_SCORES[audioLabel] ?? 0

    let pm25Score = 0
    if (pm25Val > 75) pm25Score = 2
    else if (pm25Val >= 37.5) pm25Score = 1

    let symptomScore = 0
    if (symptoms.length === 1) symptomScore = 1
    else if (symptoms.length >= 2 && symptoms.length <= 3) symptomScore = 2
    else if (symptoms.length > 3) symptomScore = 3
    if (symptoms.includes('breathing_discomfort')) symptomScore += 1

    const exposureScore = outdoor === 'more_3' ? 2 : outdoor === '1_3' ? 1 : 0
    const protectionScore = mask === 'never' ? 2 : mask === 'sometimes' ? 1 : 0
    const total = audioScore + coughPatternScore + pm25Score + symptomScore + exposureScore + protectionScore
    const level = total >= 8 ? 'high' : total >= 4 ? 'medium' : 'low'

    const symptomDetail = symptoms.length > 0
      ? `${symptoms.length} รายการ: ${getSymptomNames(symptoms).join(', ')}${symptoms.includes('breathing_discomfort') ? ' (+1 หายใจไม่สะดวก)' : ''}`
      : 'ไม่พบอาการที่ผู้ใช้เลือก'

    return {
      audioScore,
      coughPatternScore,
      pm25Score,
      symptomScore,
      exposureScore,
      protectionScore,
      total,
      level,
      maxScore: 14,
      breakdown: [
        {
          label: 'Audio Score',
          detail: coughDetected ? `${coughTypeText} ความมั่นใจ ${formatPercent(confidence)}` : AUDIO_LABEL_TEXT[audioLabel],
          points: audioScore,
        },
        {
          label: 'Cough Pattern Score',
          detail: AUDIO_LABEL_TEXT[audioLabel] || AUDIO_LABEL_TEXT.unclear,
          points: coughPatternScore,
        },
        {
          label: 'PM2.5 Score',
          detail: `${pm25Val} ${PM25_UNIT_SHORT} (${getPm25Label(pm25Val)})`,
          points: pm25Score,
        },
        {
          label: 'Symptom Score',
          detail: symptomDetail,
          points: symptomScore,
        },
        {
          label: 'Exposure Score',
          detail: OUTDOOR_LABELS[outdoor] || 'ไม่ระบุ',
          points: exposureScore,
        },
        {
          label: 'Protection Score',
          detail: MASK_LABELS[mask] || 'ไม่ระบุ',
          points: protectionScore,
        },
      ],
    }
  }, [aiResult, audioLabel, confidence, coughTypeText, mask, outdoor, pm25, symptoms])

  const associationText = useMemo(() => {
    const pm25Val = Number(pm25) || 0
    const symptomClause = symptomNames.length > 0
      ? `ร่วมกับอาการที่ผู้ใช้กรอก (${symptomNames.join(', ')})`
      : 'โดยยังไม่มีอาการร่วมที่ผู้ใช้เลือก'

    if (COUGH_AUDIO_LABELS.has(audioLabel)) {
      return `${coughTypeText} ${symptomClause} และค่า PM2.5 ${pm25Val} ${PM25_UNIT_SHORT} อาจสัมพันธ์กับการระคายคอหรือการระคายเคืองทางเดินหายใจ ควรใช้ร่วมกับอาการที่ผู้ใช้กรอกและค่า PM2.5`
    }

    if (aiResult?.possibleAssociation || aiResult?.possible_association) {
      return `${aiResult.possibleAssociation || aiResult.possible_association} ค่า PM2.5 ที่กรอกคือ ${pm25Val} ${PM25_UNIT_SHORT}`
    }

    return `ลักษณะเสียงยังไม่ชัดเจน ${symptomClause} และค่า PM2.5 ${pm25Val} ${PM25_UNIT_SHORT} ควรใช้เป็นข้อมูลคัดกรองเบื้องต้นเท่านั้น`
  }, [aiResult, audioLabel, coughTypeText, pm25, symptomNames])

  const riskLevel = RISK_LEVELS[riskData.level]
  const featureDuration = getFeatureValue(coughFeatures, 'duration_sec')
  const featureRms = getFeatureValue(coughFeatures, 'rms_level')
  const featurePeak = getFeatureValue(coughFeatures, 'peak_level')
  const featureBursts = getFeatureValue(coughFeatures, 'burst_count')

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
          <p className="section-kicker">ผลการวิเคราะห์แบบ 3 ชั้น</p>
          <h2>ผลคัดกรองเสียงไอและความเสี่ยงจากฝุ่น</h2>
          <p>สรุปลักษณะเสียงที่ระบบตรวจพบ ร่วมกับอาการที่กรอก ค่า PM2.5 เวลาอยู่กลางแจ้ง และการสวมหน้ากาก</p>
        </div>
        <img className="page-symbol result-emblem" src={logos.emblem} alt={copy.resultEmblemAlt} />
      </section>

      <section className={`ai-analysis-card ai-analysis-${audioLabel}`}>
        <div className="ai-analysis-header">
          <img src={logos.abstractMark} alt="" aria-hidden="true" />
          <div>
            <p className="section-kicker">1. AI Cough Type</p>
            <h3>AI cough analysis</h3>
          </div>
        </div>

        <div className="ai-analysis-grid">
          <div className="ai-analysis-main">
            <span>ลักษณะเสียงที่ระบบตรวจพบ</span>
            <strong>{coughTypeText}</strong>
          </div>
          <div>
            <span>Confidence</span>
            <strong>{formatPercent(confidence)}</strong>
          </div>
          <div>
            <span>Audio quality</span>
            <strong>{AUDIO_QUALITY_LABELS[audioQuality] || audioQuality || 'ไม่ระบุ'}</strong>
          </div>
        </div>

        <div className="audio-feature-grid" aria-label="รายละเอียดคุณลักษณะเสียง">
          <div>
            <span>Burst count</span>
            <strong>{featureBursts}</strong>
          </div>
          <div>
            <span>Duration</span>
            <strong>{featureDuration.toFixed(2)} s</strong>
          </div>
          <div>
            <span>RMS</span>
            <strong>{featureRms.toFixed(4)}</strong>
          </div>
          <div>
            <span>Peak</span>
            <strong>{featurePeak.toFixed(4)}</strong>
          </div>
        </div>
      </section>

      <section className="analysis-card possible-association-card card">
        <div className="summary-header">
          <div>
            <p className="section-kicker">2. Possible Symptom Association</p>
            <h3>ความสัมพันธ์ที่อาจเกี่ยวข้อง</h3>
          </div>
        </div>
        <p>{associationText}</p>
        <small>ข้อความนี้ไม่ระบุว่าผู้ใช้มีโรคใด และควรใช้ร่วมกับข้อมูลที่ผู้ใช้กรอกเท่านั้น</small>
      </section>

      <section
        className={`risk-overview risk-${riskData.level}-bg detailed-risk-card`}
        style={{ '--score-progress': `${(riskData.total / riskData.maxScore) * 360}deg` }}
      >
        <div>
          <p className="risk-label">3. Detailed Risk Score</p>
          <h3 className={`risk-${riskData.level}`}>{riskLevel.title}</h3>
          <p>{riskLevel.description}</p>
          <small className="score-context">
            Risk Score = Audio Score + Cough Pattern Score + PM2.5 Score + Symptom Score + Exposure Score + Protection Score
          </small>
        </div>
        <div className={`risk-score-display risk-score-${riskData.level}`}>
          <small>Total</small>
          <strong>{riskData.total}</strong>
          <span>{riskData.total} จาก {riskData.maxScore}</span>
        </div>

        <div className="risk-breakdown-list">
          {riskData.breakdown.map((item) => (
            <div key={item.label} className="breakdown-row">
              <div>
                <strong>{item.label}</strong>
                <span>{item.detail}</span>
              </div>
              <b className={item.points > 0 ? 'has-points' : ''}>+{item.points}</b>
            </div>
          ))}
        </div>
      </section>

      <section className={`advice-card card advice-${riskData.level}`}>
        <div className="summary-header">
          <div>
            <p className="section-kicker">4. Recommendation</p>
            <h3>คำแนะนำเบื้องต้นตามระดับความเสี่ยง</h3>
          </div>
        </div>

        {hasRedFlag && (
          <div className="red-flag-panel">
            <strong>{RED_FLAG_WARNING}</strong>
          </div>
        )}

        <div className="advice-action-grid">
          {riskLevel.recommendations.map((item, index) => (
            <article key={item} className="advice-action-card">
              <span>{index + 1}</span>
              <p>{item}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="disclaimer-panel">
        {(aiResult?.safetyNotice || aiResult?.safety_notice) && (
          <p>{aiResult.safetyNotice || aiResult.safety_notice}</p>
        )}
        <p>{SAFETY_NOTICE}</p>
      </section>

      <div className="result-actions">
        <button className="btn btn-primary btn-lg btn-full" onClick={() => navigate('/')} type="button">
          {copy.resultRestart}
        </button>
      </div>
    </div>
  )
}

export default ResultPage
