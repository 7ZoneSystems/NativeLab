import * as THREE from 'three'
import gsap from 'gsap'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { FilmPass } from 'three/examples/jsm/postprocessing/FilmPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'

export function createPostFX(renderer, scene, camera, isMobile) {
  const size = new THREE.Vector2()
  renderer.getSize(size)

  const composer = new EffectComposer(renderer)

  const renderPass = new RenderPass(scene, camera)
  composer.addPass(renderPass)

  const bloomStrength = isMobile ? 0.2 : 0.4
  const bloom = new UnrealBloomPass(
    new THREE.Vector2(size.x, size.y),
    bloomStrength,
    0.4,
    0.85
  )
  composer.addPass(bloom)

  let film = null
  if (!isMobile) {
    film = new FilmPass(0.12, 0.02, 648, false)
    composer.addPass(film)
  }

  const outputPass = new OutputPass()
  composer.addPass(outputPass)

  function resize(w, h) {
    composer.setSize(w, h)
  }

  function render() {
    composer.render()
  }

  function pulseGreenBloom() {
    gsap.to(bloom, {
      strength: 1.0,
      duration: 0.4,
      yoyo: true,
      repeat: 4,
      ease: 'sine.inOut'
    })
  }

  function setBloomStrength(val) {
    bloom.strength = val
  }

  function dispose() {
    composer.dispose()
  }

  return { composer, bloom, film, resize, render, pulseGreenBloom, setBloomStrength, dispose }
}
