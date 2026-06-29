import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'
import { scrubSection, forceShow, hideSectionLines, hideAll } from './text.js'
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
let pendingTarget = -1

// Text thresholds: { sec, scrollAt } - text reveals at this scroll %
// NOTE: s8 was given ~150vh more room than before (see body height below) so
// its lines, and especially the closing line, aren't cut off rushed. Every
// other section keeps its original absolute scroll position.
const TEXT_MAP = [
  { sec: 's1', at: 0.009 },  // "NativeLab" title
  { sec: 's2', at: 0.044 },  // "Hi, I'm Hrirake" - beam drops, extra room for subtitle
  { sec: 's3', at: 0.123 },  // "I have been using AI tools."
  { sec: 's4', at: 0.229 },  // "Your AI should belong to you."
  { sec: 's6', at: 0.405 },  // Mission statement
  { sec: 's7', at: 0.475 },  // Desktop / Android split (compressed)
  { sec: 's8', at: 0.528 },  // "not just another app..." - extended span, more breathing room
  { sec: 's9', at: 0.789 }   // CTA + logo
]

const WALK_START = 0.035
const WALK_END = 0.894
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

  document.body.style.height = '1350vh'
  buildTimeline()
}

// Move robot along spiral waypoints (never through shaft)
function moveRobotToStep(targetIndex) {
  if (!stairPositions || !avatar) return
  if (targetIndex < 0 || targetIndex >= stairPositions.length) return
  if (targetIndex === currentStep && !isMoving) return

  // If already moving, just update the target — don't kill walk
  if (isMoving) {
    pendingTarget = targetIndex
    return
  }

  isMoving = true
  pendingTarget = targetIndex
  if (avatarAnim) avatarAnim.playWalk()

  walkToTarget(targetIndex)
}

function walkToTarget(targetIndex) {
  if (walkTween) { walkTween.kill(); walkTween = null }

  const waypoints = []
  const dir = targetIndex > currentStep ? 1 : -1
  for (let i = currentStep; i !== targetIndex; i += dir) {
    const s = stairPositions[i]
    waypoints.push({ x: s.x, y: s.y - feetOffset, z: s.z })
  }
  const final = stairPositions[targetIndex]
  waypoints.push({ x: final.x, y: final.y - feetOffset, z: final.z })

  // Scale speed: more steps = faster per step, clamped
  const stepCount = Math.abs(targetIndex - currentStep)
  const perStepDuration = Math.max(0.08, Math.min(0.3, 0.6 / stepCount))

  // Face first direction
  if (waypoints.length > 1) {
    const dx = waypoints[1].x - waypoints[0].x
    const dz = waypoints[1].z - waypoints[0].z
    if (Math.abs(dx) > 0.001 || Math.abs(dz) > 0.001) {
      gsap.to(avatar.rotation, { y: Math.atan2(dx, dz), duration: 0.15, ease: 'power2.out' })
    }
  }

  let wpIdx = 0
  function nextWP() {
    // Check if a new target was set while walking
    if (pendingTarget !== targetIndex && pendingTarget !== currentStep) {
      // Redirect to new target without stopping
      currentStep = targetIndex
      walkToTarget(pendingTarget)
      return
    }

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
      duration: perStepDuration, ease: 'none',
      onComplete: nextWP
    })

    if (wpIdx < waypoints.length) {
      const next = waypoints[wpIdx]
      const dx = next.x - wp.x
      const dz = next.z - wp.z
      if (Math.abs(dx) > 0.001 || Math.abs(dz) > 0.001) {
        gsap.to(avatar.rotation, { y: Math.atan2(dx, dz), duration: 0.1 })
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
        // Fade out previous section (animated, no abrupt cut)
        if (prevSection) hideSectionLines(prevSection)
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

      // ── FEATURE TAGS: show during 33.4-40.5% (same absolute scroll spot as before) ─
      if (sp >= 0.334 && sp <= 0.405) {
        const featProgress = (sp - 0.334) / 0.071
        const featIdx = Math.floor(featProgress * 8)
        for (let i = 0; i < 8; i++) {
          if (i <= featIdx) showFeatureTag(i)
        }
      } else {
        hideAllFeatureTags()
      }

      // ── TIE GLOW: fire at same absolute spot as before (now 86.8%) ───
      if (sp >= 0.868 && avatarAnim) {
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
        if (targetStep !== prevStep) {
          moveRobotToStep(targetStep)
          prevStep = targetStep
        }
      }
    }
  })
}