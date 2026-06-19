import * as THREE from 'three'
import gsap from 'gsap'

const SVG_ICONS = {
  brain: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a4 4 0 0 1 4 4c0 .73-.2 1.41-.54 2A4 4 0 0 1 18 12c0 1.1-.45 2.1-1.17 2.83A4 4 0 0 1 14 22h-4a4 4 0 0 1-2.83-7.17A4 4 0 0 1 6 12a4 4 0 0 1 2.54-3.72A4 4 0 0 1 8 6a4 4 0 0 1 4-4z"/><path d="M12 2v20"/></svg>`,
  folder: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
  image: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>`,
  server: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="8" rx="2" ry="2"/><rect x="2" y="14" width="20" height="8" rx="2" ry="2"/><line x1="6" y1="6" x2="6.01" y2="6"/><line x1="6" y1="18" x2="6.01" y2="18"/></svg>`,
  phone: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="2" width="14" height="20" rx="2" ry="2"/><line x1="12" y1="18" x2="12.01" y2="18"/></svg>`,
  wrench: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>`,
  lock: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>`,
  zap: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`
}

const FEATURES = [
  { id: 'feat1', icon: 'brain', title: 'On-Device Inference', detail: 'Runs entirely on your hardware. No cloud. No latency.' },
  { id: 'feat2', icon: 'folder', title: 'Document RAG', detail: 'PDF, TXT, DOCX. Ask questions across your files.' },
  { id: 'feat3', icon: 'image', title: 'Vision Models', detail: 'Multimodal. Send images. Get answers.' },
  { id: 'feat4', icon: 'server', title: 'Local API Server', detail: 'OpenAI-compatible on your network.' },
  { id: 'feat5', icon: 'phone', title: 'PhonoLab Android', detail: 'Full inference engine in your pocket.' },
  { id: 'feat6', icon: 'wrench', title: 'Model Management', detail: 'Download, quantize, switch models instantly.' },
  { id: 'feat7', icon: 'lock', title: 'Privacy First', detail: 'Your data never leaves your machine. Ever.' },
  { id: 'feat8', icon: 'zap', title: 'Pipeline Builder', detail: 'Visual multi-step AI workflows.' }
]

let tagElements = []

export function initFeatures(isMobile) {
  if (isMobile) {
    initMobileFeatures()
    return
  }

  const container = document.getElementById('feature-tags-container')
  if (!container) return

  FEATURES.forEach(feat => {
    const tag = document.createElement('div')
    tag.className = 'feature-tag'
    tag.id = feat.id
    tag.innerHTML = `
      <div class="feature-tag-icon">${SVG_ICONS[feat.icon]}</div>
      <h3>${feat.title}</h3>
      <p>${feat.detail}</p>
    `
    container.appendChild(tag)
    tagElements.push(tag)
  })
}

function initMobileFeatures() {
  const container = document.getElementById('mobile-features')
  if (!container) return

  FEATURES.forEach(feat => {
    const card = document.createElement('div')
    card.className = 'mobile-feature-card'
    card.innerHTML = `
      <div class="mfc-icon">${SVG_ICONS[feat.icon]}</div>
      <h3>${feat.title}</h3>
      <p>${feat.detail}</p>
    `
    container.appendChild(card)
  })

  // IntersectionObserver for mobile cards
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view')
        observer.unobserve(entry.target)
      }
    })
  }, { threshold: 0.2 })

  container.querySelectorAll('.mobile-feature-card').forEach(el => observer.observe(el))
}

export function showFeatureTag(index) {
  if (index < 0 || index >= tagElements.length) return
  const tag = tagElements[index]
  tag.classList.add('visible')
  gsap.from(tag, {
    x: -40,
    opacity: 0,
    duration: 0.5,
    ease: 'power2.out'
  })
}

export function hideFeatureTag(index) {
  if (index < 0 || index >= tagElements.length) return
  const tag = tagElements[index]
  gsap.to(tag, {
    opacity: 0,
    x: -20,
    duration: 0.3,
    ease: 'power2.in',
    onComplete: () => tag.classList.remove('visible')
  })
}

export function hideAllFeatureTags() {
  tagElements.forEach((tag, i) => hideFeatureTag(i))
}

export function positionFeatureTag(index, screenPos) {
  if (index < 0 || index >= tagElements.length) return
  const tag = tagElements[index]
  tag.style.top = `${screenPos.y}px`
}
