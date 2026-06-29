import gsap from 'gsap'

// Each section's lines, in order
const SECTIONS = {
  s1: ['text-s1-title', 'text-s1-sub'],
  s2: ['text-s2a', 'text-s2b'],
  s3: ['text-s3a', 'text-s3b', 'text-s3c', 'text-s3d'],
  s4: ['text-s4a', 'text-s4b', 'text-s4c', 'text-s4d', 'text-s4e'],
  s6: ['text-s6a', 'text-s6b', 'text-s6c'],
  s7: ['text-s7'],
  s8: ['text-s8a', 'text-s8b', 'text-s8c', 'text-s8d'],
  s9: ['cta-title', 'cta-sub', 'cta-form', 'cta-social', 'cta-footer']
}

// Track which lines are currently visible
const lineStates = {}

function initLineStates() {
  for (const [sec, lines] of Object.entries(SECTIONS)) {
    lines.forEach(id => {
      lineStates[id] = false
      const el = document.getElementById(id)
      if (el) {
        el.style.opacity = '0'
        el.style.transform = 'translateY(20px)'
        el.style.transition = 'none'
      }
    })
    // Also hide the parent block
    const blockId = `text-${sec}`
    const block = document.getElementById(blockId)
    if (block) {
      block.style.opacity = '0'
      block.style.transition = 'none'
    }
  }
}

export function initText() {
  initLineStates()
}

// Show a specific line with animation
function showLine(id) {
  if (lineStates[id]) return
  lineStates[id] = true
  const el = document.getElementById(id)
  if (!el) return

  // Show parent block
  const block = el.closest('.text-block') || el.closest('.split-container')
  if (block) {
    block.style.opacity = '1'
  }

  gsap.to(el, {
    opacity: 1,
    y: 0,
    duration: 0.5,
    ease: 'power2.out'
  })
}

// Hide a specific line
function hideLine(id) {
  if (!lineStates[id]) return
  lineStates[id] = false
  const el = document.getElementById(id)
  if (!el) return

  gsap.to(el, {
    opacity: 0,
    y: 20,
    duration: 0.3,
    ease: 'power2.in'
  })
}

// Hide all lines in a section (animated fade-out — use this for section
// transitions instead of forceHide, which snaps instantly with no animation)
export function hideSectionLines(sec) {
  const lines = SECTIONS[sec]
  if (!lines) return
  lines.forEach(id => hideLine(id))

  // Hide parent block after all lines hidden
  setTimeout(() => {
    const block = document.getElementById(`text-${sec}`)
    if (block) {
      const anyVisible = lines.some(id => lineStates[id])
      if (!anyVisible) block.style.opacity = '0'
    }
  }, 350)
}

// Scrub a section's text based on progress (0-1)
// Each line reveals at equal intervals within the section
// Returns: true when ALL lines in section are revealed
export function scrubSection(sec, progress) {
  const lines = SECTIONS[sec]
  if (!lines) return true

  const lineCount = lines.length
  const block = document.getElementById(`text-${sec}`)
  if (block && progress > 0) block.style.opacity = '1'

  for (let i = 0; i < lineCount; i++) {
    const lineThreshold = (i + 1) / (lineCount + 1) // stagger within section
    if (progress >= lineThreshold) {
      showLine(lines[i])
    } else {
      hideLine(lines[i])
    }
  }

  // All revealed when progress > last threshold
  return progress >= (lineCount / (lineCount + 1))
}

// Reverse scrub - hide lines from last to first
export function scrubSectionReverse(sec, progress) {
  const lines = SECTIONS[sec]
  if (!lines) return

  const lineCount = lines.length

  for (let i = lineCount - 1; i >= 0; i--) {
    const lineThreshold = (i + 0.5) / (lineCount + 1)
    if (progress < lineThreshold) {
      hideLine(lines[i])
    }
  }

  const block = document.getElementById(`text-${sec}`)
  if (block && progress <= 0) {
    block.style.opacity = '0'
  }
}

// Check if all lines in a section are visible
export function isSectionComplete(sec) {
  const lines = SECTIONS[sec]
  if (!lines) return true
  return lines.every(id => lineStates[id])
}

// Force show/hide for immediate transitions
export function forceShow(sec) {
  const lines = SECTIONS[sec]
  if (!lines) return
  const block = document.getElementById(`text-${sec}`)
  if (block) block.style.opacity = '1'
  lines.forEach(id => {
    lineStates[id] = true
    const el = document.getElementById(id)
    if (el) {
      el.style.opacity = '1'
      el.style.transform = 'translateY(0)'
    }
  })
}

export function forceHide(sec) {
  const lines = SECTIONS[sec]
  if (!lines) return
  lines.forEach(id => {
    lineStates[id] = false
    const el = document.getElementById(id)
    if (el) {
      el.style.opacity = '0'
      el.style.transform = 'translateY(20px)'
    }
  })
  const block = document.getElementById(`text-${sec}`)
  if (block) block.style.opacity = '0'
}

export function hideAll() {
  for (const sec of Object.keys(SECTIONS)) {
    forceHide(sec)
  }
}