import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { scrubSection, forceShow, forceHide, hideAll } from './text.js'
import { showFeatureTag, hideAllFeatureTags } from './features.js'

gsap.registerPlugin(ScrollTrigger)

let cameraFollow = null
let avatar = null
let avatarAnim = null
let stairPositions = null
let postfx = null
let navEl = null
let feetOffset = 0
let activateBeam = null

let currentStep = 0
let isMoving = false
let walkTween = null
let beamFired = false

// Text thresholds: { sec, scrollAt } - text reveals at this scroll %
const TEXT_MAP = [
  { sec: 's1', at: 0.01 },   // "NativeLab" title
  { sec: 's2', at: 0.05 },   // "Hi, I'm Hrirake" - beam drops, extra room for subtitle
  { sec: 's3', at: 0.14 },   // "I have been using AI tools."
  { sec: 's4', at: 0.26 },   // "Your AI should belong to you."
  { sec: 's6', at: 0.46 },   // Mission statement
  { sec: 's7', at: 0.54 },   // Desktop / Android split (compressed)
  { sec: 's8', at: 0.60 },   // "free. fast. yours." (closer to s6)
  { sec: 's9', at: 0.76 }    // CTA + logo
]

const WALK_START = 0.04
const WALK_END = 0.88
const TOTAL_STEPS = 40

export function initScroll(opts) {
  cameraFollow = opts.cameraFollow
  avatar = opts.avatar
  avatarAnim = opts.avatarAnim
  stairPositions = opts.stairPositions
  postfx = opts.postfx
  navEl = document.getElementById('site-nav')
  feetOffset = opts.feetOffset || 0
  activateBeam = opts.activateBeam

  document.body.style.height = '1200vh'
  buildTimeline()
}

// Move robot along spiral waypoints (never through shaft)
function moveRobotToStep(targetIndex) {
  if (!stairPositions || !avatar) return
  if (targetIndex < 0 || targetIndex >= stairPositions.length) return
  if (targetIndex === currentStep || isMoving) return

  if (walkTween) { walkTween.kill(); walkTween = null }
  isMoving = true
  if (avatarAnim) avatarAnim.playWalk()

  const waypoints = []
  const dir = targetIndex > currentStep ? 1 : -1
  for (let i = currentStep; i !== targetIndex; i += dir) {
    const s = stairPositions[i]
    waypoints.push({ x: s.x, y: s.y - feetOffset, z: s.z })
  }
  const final = stairPositions[targetIndex]
  waypoints.push({ x: final.x, y: final.y - feetOffset, z: final.z })

  // Face first direction
  if (waypoints.length > 1) {
    const dx = waypoints[1].x - waypoints[0].x
    const dz = waypoints[1].z - waypoints[0].z
    if (Math.abs(dx) > 0.001 || Math.abs(dz) > 0.001) {
      gsap.to(avatar.rotation, { y: Math.atan2(dx, dz), duration: 0.2, ease: 'power2.out' })
    }
  }

  let wpIdx = 0
  function nextWP() {
    if (wpIdx >= waypoints.length) {
      isMoving = false
      currentStep = targetIndex
      if (avatarAnim) avatarAnim.playIdle()
      return
    }
    const wp = waypoints[wpIdx]
    wpIdx++

    walkTween = gsap.to(avatar.position, {
      x: wp.x, y: wp.y, z: wp.z,
      duration: 0.35, ease: 'none',
      onComplete: nextWP
    })

    if (wpIdx < waypoints.length) {
      const next = waypoints[wpIdx]
      const dx = next.x - wp.x
      const dz = next.z - wp.z
      if (Math.abs(dx) > 0.001 || Math.abs(dz) > 0.001) {
        gsap.to(avatar.rotation, { y: Math.atan2(dx, dz), duration: 0.15 })
      }
    }
  }
  nextWP()
}

// Determine which text section should be active at this scroll position
function getActiveSection(scrollProgress) {
  let active = null
  for (const entry of TEXT_MAP) {
    if (scrollProgress >= entry.at) {
      active = entry.sec
    }
  }
  return active
}

function buildTimeline() {
  // Nav
  ScrollTrigger.create({
    trigger: document.body, start: 'top top', end: '5% top',
    onEnter: () => navEl && navEl.classList.add('visible'),
    onLeaveBack: () => navEl && navEl.classList.remove('visible')
  })

  // ═══ MASTER SCROLL - drives everything ═══════════════════════
  let prevSection = null
  let prevStep = -1

  ScrollTrigger.create({
    trigger: document.body,
    start: 'top top',
    end: 'bottom bottom',
    scrub: 0.3,
    onUpdate: (self) => {
      const sp = self.progress

      // ── TEXT: determine active section, switch if changed ─────
      const activeSec = getActiveSection(sp)
      if (activeSec !== prevSection) {
        // Fade out previous section
        if (prevSection) forceHide(prevSection)
        prevSection = activeSec
      }

      // Scrub active section's text based on local progress
      if (activeSec) {
        const entry = TEXT_MAP.find(e => e.sec === activeSec)
        const nextEntry = TEXT_MAP.find(e => e.at > entry.at)
        const secEnd = nextEntry ? nextEntry.at : 1.0
        const secProgress = Math.min(1, (sp - entry.at) / (secEnd - entry.at))
        scrubSection(activeSec, secProgress)
      }

      // ── BEAM: fire at 5% ─────────────────────────────────────
      if (sp >= 0.05 && !beamFired) {
        beamFired = true
        if (activateBeam) activateBeam()
      }

      // ── FEATURE TAGS: show during 38-46% ─────────────────────
      if (sp >= 0.38 && sp <= 0.46) {
        const featProgress = (sp - 0.38) / 0.08
        const featIdx = Math.floor(featProgress * 8)
        for (let i = 0; i < 8; i++) {
          if (i <= featIdx) showFeatureTag(i)
        }
      } else {
        hideAllFeatureTags()
      }

      // ── TIE GLOW: fire at 85% ────────────────────────────────
      if (sp >= 0.85 && avatarAnim) {
        avatarAnim.glowPatch()
        if (postfx) postfx.pulseGreenBloom()
        avatarAnim = null // only fire once
      }

      // ── WALK: continuous robot movement ───────────────────────
      if (sp >= WALK_START && sp <= WALK_END) {
        const walkProgress = (sp - WALK_START) / (WALK_END - WALK_START)
        const targetStep = Math.min(
          TOTAL_STEPS - 1,
          Math.floor(walkProgress * TOTAL_STEPS)
        )
        if (targetStep !== prevStep && targetStep !== currentStep && !isMoving) {
          moveRobotToStep(targetStep)
          prevStep = targetStep
        }
      }
    }
  })
}
