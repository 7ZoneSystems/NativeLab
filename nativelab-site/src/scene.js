import * as THREE from 'three'
import gsap from 'gsap'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'

export function createScene() {
  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false,
    powerPreference: 'high-performance'
  })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.shadowMap.enabled = true
  renderer.shadowMap.type = THREE.PCFSoftShadowMap
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.1
  renderer.outputColorSpace = THREE.SRGBColorSpace

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x010101)
  scene.fog = new THREE.FogExp2(0x010101, 0.035)

  const camera = new THREE.PerspectiveCamera(
    55,
    window.innerWidth / window.innerHeight,
    0.1,
    100
  )
  camera.position.set(0, 10, 14)

  // ── ENVIRONMENT MAP for reflections ──────────────────────────
  // Dim environment so metallic surfaces have something to reflect
  const pmremGenerator = new THREE.PMREMGenerator(renderer)
  pmremGenerator.compileEquirectangularShader()
  const envScene = new RoomEnvironment()
  const envTexture = pmremGenerator.fromScene(envScene, 0.04).texture
  scene.environment = envTexture // all MeshStandardMaterial will use this
  envScene.dispose()
  pmremGenerator.dispose()

  // ── ALL LIGHTS START AT ZERO ─────────────────────────────────
  const ambient = new THREE.AmbientLight(0xffffff, 0.0)
  scene.add(ambient)

  const dirLight = new THREE.DirectionalLight(0xffffff, 0.0)
  dirLight.position.set(-5, 15, 5)
  dirLight.castShadow = true
  dirLight.shadow.mapSize.set(2048, 2048)
  dirLight.shadow.camera.near = 0.1
  dirLight.shadow.camera.far = 50
  dirLight.shadow.camera.left = -12
  dirLight.shadow.camera.right = 12
  dirLight.shadow.camera.top = 15
  dirLight.shadow.camera.bottom = -15
  dirLight.shadow.bias = -0.001
  dirLight.shadow.normalBias = 0.02
  scene.add(dirLight)

  // ── BEAM FIXTURE ─────────────────────────────────────────────
  const beamFixture = new THREE.Group()
  beamFixture.position.set(0, 35, 0)

  // Housing
  const housing = new THREE.Mesh(
    new THREE.CylinderGeometry(0.3, 0.4, 0.5, 16),
    new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.3, metalness: 0.8 })
  )
  beamFixture.add(housing)

  // Rim
  const rim = new THREE.Mesh(
    new THREE.TorusGeometry(0.42, 0.04, 8, 24),
    new THREE.MeshStandardMaterial({ color: 0x888888, roughness: 0.2, metalness: 0.9 })
  )
  rim.rotation.x = Math.PI / 2
  rim.position.y = -0.25
  beamFixture.add(rim)

  // Bulb
  const bulbMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    emissive: 0xffffff,
    emissiveIntensity: 0,
    roughness: 0.1
  })
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.15, 16, 16), bulbMat)
  bulb.position.y = -0.15
  beamFixture.add(bulb)

  // Cord
  const cord = new THREE.Mesh(
    new THREE.CylinderGeometry(0.015, 0.015, 20, 4),
    new THREE.MeshBasicMaterial({ color: 0x111111 })
  )
  cord.position.y = 10
  beamFixture.add(cord)

  // Volumetric cone - very subtle
  const coneMat = new THREE.MeshBasicMaterial({
    color: 0xffeedd,
    transparent: true,
    opacity: 0.0,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  })
  const cone = new THREE.Mesh(new THREE.ConeGeometry(2.5, 12, 32, 1, true), coneMat)
  cone.position.y = -6.5
  beamFixture.add(cone)

  scene.add(beamFixture)

  // ── SPOTLIGHT - realistic intensity ──────────────────────────
  const beamLight = new THREE.SpotLight(0xffffff, 0.0)
  beamLight.position.set(0, 15, 0)
  beamLight.angle = Math.PI * 0.08
  beamLight.penumbra = 0.6
  beamLight.decay = 1.8
  beamLight.distance = 30
  beamLight.castShadow = true
  beamLight.shadow.mapSize.set(1024, 1024)
  beamLight.shadow.bias = -0.001
  scene.add(beamLight)
  scene.add(beamLight.target)

  // Subtle fill from side - picks up on nearby stairs/shaft
  const fillLight = new THREE.PointLight(0x334466, 0.0, 18)
  fillLight.position.set(5, 6, -4)
  scene.add(fillLight)

  // Warm bounce from floor - subtle
  const bounceLight = new THREE.PointLight(0x997744, 0.0, 12)
  bounceLight.position.set(-2, -3, 3)
  scene.add(bounceLight)

  // Ground
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(60, 60),
    new THREE.MeshStandardMaterial({ color: 0x030303, roughness: 1, metalness: 0 })
  )
  ground.rotation.x = -Math.PI / 2
  ground.position.y = -7.5
  ground.receiveShadow = true
  scene.add(ground)

  // ── BEAM CONTROL ─────────────────────────────────────────────
  let beamState = 'off'
  let beamIntensity = 0

  function activateBeam() {
    if (beamState !== 'off') return
    beamState = 'dropping'

    gsap.to(beamFixture.position, {
      y: 14,
      duration: 1.2,
      ease: 'power2.out',
      onComplete: () => { beamState = 'on' }
    })

    gsap.to(bulbMat, {
      emissiveIntensity: 1.5,
      duration: 1.0,
      delay: 0.3
    })
  }

  function updateBeam(delta, avatarPos) {
    if (beamState === 'dropping' || beamState === 'on') {
      beamIntensity = Math.min(1, beamIntensity + delta * 0.8)
    }

    const t = beamIntensity
    const fixtureY = beamFixture.position.y

    // Follow robot
    if (avatarPos && beamState !== 'off') {
      beamLight.position.set(avatarPos.x, fixtureY - 1, avatarPos.z)
      beamLight.target.position.set(avatarPos.x, avatarPos.y + 0.5, avatarPos.z)
      beamFixture.position.x = avatarPos.x
      beamFixture.position.z = avatarPos.z
    }

    // REALISTIC intensities - not blinding
    beamLight.intensity = t * 3.0          // spotlight on robot + nearby
    coneMat.opacity = t * 0.04             // very faint volumetric cone
    fillLight.intensity = t * 0.08         // subtle blue fill
    bounceLight.intensity = t * 0.06       // subtle warm bounce
    ambient.intensity = t * 0.015          // barely there ambient
    dirLight.intensity = t * 0.15          // faint directional for shadows
    renderer.toneMappingExposure = 0.1 + (t * 0.4)
  }

  return {
    renderer, scene, camera,
    activateBeam,
    updateBeam
  }
}
