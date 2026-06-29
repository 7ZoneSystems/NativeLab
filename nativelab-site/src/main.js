import * as THREE from 'three'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

import { createScene } from './scene.js'
import { createStairs } from './stairs.js'
import { createAvatar } from './avatar.js'
import { CameraFollow } from './camera.js'
import { createPostFX } from './postfx.js'
import { initScroll } from './scroll.js'
import { initText, hideAll, forceShow } from './text.js'
import { initFeatures } from './features.js'
import { isMobile } from './mobile.js'

gsap.registerPlugin(ScrollTrigger)

const loadingScreen = document.getElementById('loading-screen')
const loadingBar = document.getElementById('loading-bar')
const loadingPct = document.getElementById('loading-pct')

function setProgress(pct) {
  const c = Math.min(100, Math.max(0, Math.round(pct)))
  if (loadingBar) loadingBar.style.width = c + '%'
  if (loadingPct) loadingPct.textContent = c + '%'
}

function yieldFrame() {
  return new Promise(resolve => requestAnimationFrame(resolve))
}

function dismissLoading(immediate = false) {
  if (!loadingScreen) return
  gsap.to(loadingScreen, {
    opacity: 0,
    duration: immediate ? 0 : 0.8,
    delay: immediate ? 0 : 0.15,
    onComplete: () => loadingScreen.remove()
  })
}

function showLoadingError(message) {
  if (!loadingScreen) return
  const info = loadingScreen.querySelector('.loader-info')
  if (info) {
    info.innerHTML = `<span class="loader-error">${message}</span>`
  }
  if (loadingBar) {
    loadingBar.style.width = '100%'
    loadingBar.style.background = '#f36f6f'
  }
  if (loadingPct) {
    loadingPct.textContent = 'Error'
  }
  loadingScreen.classList.add('error')
}

async function init() {
  const { renderer, scene, camera, activateBeam, updateBeam } = createScene()
  const container = document.getElementById('canvas-container')
  container.appendChild(renderer.domElement)
  setProgress(25)
  await yieldFrame()

  const { stairPositions, shaft } = createStairs(scene)
  setProgress(45)
  await yieldFrame()

  const avatarData = await createAvatar((p) => {
    if (p.total) setProgress(45 + Math.round((p.loaded / p.total) * 25))
  })

  const { avatar, mixer, playIdle, playWalk, glowPatch, feetOffset } = avatarData
  scene.add(avatar)

  // Place avatar on first step
  if (stairPositions.length > 0) {
    const first = stairPositions[0]
    avatar.position.set(first.x, first.y - feetOffset, first.z)
    avatar.rotation.y = first.facingY
  }

  setProgress(75)
  await yieldFrame()

  // Pre-compile shaders (avoids stutter on first render)
  renderer.compile(scene, camera)
  renderer.render(scene, camera)

  const cameraFollow = new CameraFollow(camera)
  setProgress(80)
  await yieldFrame()

  let postfx = null
  try { postfx = createPostFX(renderer, scene, camera, isMobile) } catch (e) { console.warn('PostFX:', e) }
  setProgress(88)
  await yieldFrame()

  initText()
  hideAll()
  forceShow('s1')
  initFeatures(isMobile)

  initScroll({
    cameraFollow, avatar,
    avatarAnim: { playIdle, playWalk, glowPatch },
    stairPositions, postfx, feetOffset,
    activateBeam
  })

  setProgress(100)
  await yieldFrame()
  dismissLoading()

  const clock = new THREE.Clock()

  function animate() {
    requestAnimationFrame(animate)
    const delta = Math.min(clock.getDelta(), 0.05)
    if (document.hidden) return

    if (mixer) mixer.update(delta)

    const sp = window.scrollY / (document.body.scrollHeight - window.innerHeight)
    cameraFollow.update(delta, avatar.position, sp)

    // Update beam spotlight to follow robot
    updateBeam(delta, avatar.position)

    if (postfx) postfx.render()
    else renderer.render(scene, camera)
  }
  animate()

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
    if (postfx) postfx.resize(window.innerWidth, window.innerHeight)
  })

  gsap.ticker.lagSmoothing(0)
}

init().catch(err => {
  console.error('Scene init failed:', err)
  const message = err?.message?.includes('WebGL')
    ? 'WebGL is unavailable. Please enable hardware acceleration or try a different browser.'
    : 'An unexpected error occurred while loading the page. Please refresh.'
  showLoadingError(message)
  setProgress(100)
  dismissLoading(true)
})
