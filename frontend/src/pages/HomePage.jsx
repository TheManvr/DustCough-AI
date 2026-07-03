import { useNavigate } from 'react-router-dom'
import { copy } from '../assets/copy'
import { logos } from '../assets/logos'
import './HomePage.css'

const FLOW_STEPS = [
  {
    number: '1',
    title: copy.homeFlow1Title,
    text: copy.homeFlow1Text,
    featured: true,
  },
  {
    number: '2',
    title: copy.homeFlow2Title,
    text: copy.homeFlow2Text,
    featured: true,
  },
  {
    number: '3',
    title: copy.homeFlow3Title,
    text: copy.homeFlow3Text,
  },
  {
    number: '4',
    title: copy.homeFlow4Title,
    text: copy.homeFlow4Text,
  },
]

const HERO_WAVE_BARS = [
  0.32, 0.48, 0.7, 0.42, 0.58, 0.86, 0.5, 0.76,
  0.38, 0.62, 0.92, 0.56, 0.78, 0.44, 0.66, 0.84,
  0.52, 0.72, 0.4, 0.6, 0.88, 0.54, 0.74, 0.46,
]

function HomePage() {
  const navigate = useNavigate()

  return (
    <div className="home-page">
      <section className="intro-panel mvp-hero">
        <div className="intro-copy">
          <img className="home-logo" src={logos.combination} alt="โลโก้ DustCough AI" />
          <div className="status-label">PM2.5 Health Risk Awareness | Lamphun</div>
          <h2>
            วิเคราะห์<span className="gradient-text">เสียงไอด้วย AI</span> เพื่อคัดกรองความเสี่ยงทางเดินหายใจเบื้องต้น
          </h2>
          <p>{copy.homeAiHeroDescription}</p>

          <button
            className="btn btn-primary btn-lg btn-full"
            onClick={() => navigate('/record')}
            type="button"
          >
            {copy.homeStartCoughAnalysis}
          </button>

          <div className="diagnosis-note">ไม่ใช่เครื่องมือวินิจฉัยโรค</div>
        </div>

        <div className="hero-media audio-visual" aria-label="ภาพจำลองการวิเคราะห์เสียงไอด้วย AI">
          <div className="audio-orb" aria-hidden="true">
            <span className="audio-ring ring-one" />
            <span className="audio-ring ring-two" />
            <span className="audio-ring ring-three" />
            <img className="audio-symbol" src={logos.symbol} alt="" />
          </div>

          <div className="hero-waveform" aria-hidden="true">
            {HERO_WAVE_BARS.map((level, index) => (
              <span key={`${level}-${index}`} style={{ '--bar-index': index, '--bar-level': level }} />
            ))}
          </div>
        </div>
      </section>

      <section className="problem-card">
        <p className="section-kicker">โจทย์ของพื้นที่</p>
        <h3>ลำพูนและภาคเหนือมีช่วงที่ค่าฝุ่น PM2.5 สูงจากหมอกควันและการสะสมของฝุ่นละเอียด</h3>
        <p>
          ผู้ใช้จำนวนมากเห็นเพียงตัวเลขค่าฝุ่น แต่ยังไม่รู้ว่าอาการของตนเองควรเฝ้าระวังมากน้อยเพียงใด
          MVP นี้จึงออกแบบให้เข้าใจง่าย ใช้เร็ว และช่วยตัดสินใจดูแลตนเองได้ดีขึ้น
        </p>
      </section>

      <section className="screening-summary flow-grid" aria-label="ขั้นตอนการใช้งาน DustCough AI">
        {FLOW_STEPS.map((step, index) => (
          <article className={`summary-item flow-item ${step.featured ? 'flow-item-featured' : ''}`} key={step.number}>
            <span className="summary-number">{step.number}</span>
            <div>
              <h3>{step.title}</h3>
              <p>{step.text}</p>
            </div>
            {index < FLOW_STEPS.length - 1 && <span className="flow-arrow" aria-hidden="true" />}
          </article>
        ))}
      </section>
    </div>
  )
}

export default HomePage
