import * as THREE from 'three'

const SHAFT_R = 1.8
const STEP_COUNT = 40
const REVOLUTIONS = 3.0
const TOTAL_DROP = 7.0
const STEP_H = TOTAL_DROP / STEP_COUNT
const STEP_W = 1.3
const STEP_D = 0.38
const STAND_R = SHAFT_R + STEP_W * 0.5 + 0.4
const APR = (Math.PI * 2 * REVOLUTIONS) / STEP_COUNT
const START_Y = 6.5

export function createStairs(scene) {
  const stairPositions = []

  console.log('=== STAIR CONFIG ===')
  console.log('Shaft radius:', SHAFT_R)
  console.log('Step stand radius:', STAND_R, '← must be >', SHAFT_R)
  console.log('Step height:', STEP_H.toFixed(4))
  console.log('Step count:', STEP_COUNT)
  console.log('Angle per step:', (APR * 180 / Math.PI).toFixed(1), 'deg')

  // ── SHAFT ────────────────────────────────────────────────────
  const shaftCanvas = document.createElement('canvas')
  shaftCanvas.width = shaftCanvas.height = 256
  const sctx = shaftCanvas.getContext('2d')
  sctx.fillStyle = '#141414'
  sctx.fillRect(0, 0, 256, 256)
  for (let i = 0; i < 3000; i++) {
    const v = Math.random() > 0.5 ? 30 : 10
    sctx.fillStyle = `rgba(${v},${v},${v},0.3)`
    sctx.fillRect(Math.random() * 256, Math.random() * 256, 2, 2)
  }
  const shaftTex = new THREE.CanvasTexture(shaftCanvas)
  shaftTex.wrapS = shaftTex.wrapT = THREE.RepeatWrapping
  shaftTex.repeat.set(4, 8)

  const shaft = new THREE.Mesh(
    new THREE.CylinderGeometry(SHAFT_R, SHAFT_R, 50, 64),
    new THREE.MeshStandardMaterial({
      map: shaftTex, color: 0x141414, roughness: 0.9, metalness: 0.15
    })
  )
  shaft.position.y = 0
  shaft.castShadow = true
  shaft.receiveShadow = true
  shaft.name = 'shaft'
  scene.add(shaft)

  // Shaft rings
  const ringGeo = new THREE.TorusGeometry(SHAFT_R + 0.02, 0.03, 8, 64)
  const ringMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.5, metalness: 0.4 })
  for (let i = 0; i < 8; i++) {
    const ring = new THREE.Mesh(ringGeo, ringMat)
    ring.position.y = -16 + i * 4.5
    ring.rotation.x = Math.PI / 2
    ring.castShadow = true
    ring.receiveShadow = true
    scene.add(ring)
  }

  // ── MATERIALS ────────────────────────────────────────────────
  const treadMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0x2a2a2a),
    roughness: 0.92,
    metalness: 0.02,
  })
  const riserMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0x1c1c1c),
    roughness: 0.95,
    metalness: 0.0,
  })
  const railMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0x4a4a4a),
    roughness: 0.35,
    metalness: 0.75,
  })

  // ── BUILD 40 CONNECTED STEPS ─────────────────────────────────
  for (let i = 0; i < STEP_COUNT; i++) {
    const angle = i * APR
    const y = START_Y - (i * STEP_H)
    const cx = Math.cos(angle) * STAND_R
    const cz = Math.sin(angle) * STAND_R

    // Tread
    const tread = new THREE.Mesh(
      new THREE.BoxGeometry(STEP_W, 0.045, STEP_D),
      treadMat
    )
    tread.castShadow = true
    tread.receiveShadow = true

    // Riser - vertical face at front of tread
    const riser = new THREE.Mesh(
      new THREE.BoxGeometry(STEP_W, STEP_H, 0.04),
      riserMat
    )
    riser.position.set(0, -STEP_H / 2, STEP_D / 2 - 0.02)
    riser.castShadow = true
    riser.receiveShadow = true

    // Step group
    const sg = new THREE.Group()
    sg.add(tread)
    sg.add(riser)
    sg.position.set(cx, y, cz)
    sg.rotation.y = -angle + Math.PI / 2
    scene.add(sg)

    stairPositions.push({
      x: cx,
      y: y + 0.045,
      z: cz,
      facingY: -angle + Math.PI / 2 + Math.PI
    })
  }

  // ── OUTER RAILING (solid tube) ───────────────────────────────
  const outerPts = []
  for (let i = -1; i <= STEP_COUNT + 1; i++) {
    const a = i * APR
    const r = STAND_R + STEP_W / 2 - 0.05
    outerPts.push(new THREE.Vector3(
      Math.cos(a) * r,
      START_Y - (i * STEP_H) + 0.85,
      Math.sin(a) * r
    ))
  }
  const outerRail = new THREE.Mesh(
    new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(outerPts),
      STEP_COUNT * 8, 0.032, 8, false
    ),
    railMat
  )
  outerRail.castShadow = true
  scene.add(outerRail)

  // ── INNER RAILING ────────────────────────────────────────────
  const innerPts = []
  for (let i = -1; i <= STEP_COUNT + 1; i++) {
    const a = i * APR
    const r = STAND_R - STEP_W / 2 + 0.05
    innerPts.push(new THREE.Vector3(
      Math.cos(a) * r,
      START_Y - (i * STEP_H) + 0.85,
      Math.sin(a) * r
    ))
  }
  const innerRail = new THREE.Mesh(
    new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(innerPts),
      STEP_COUNT * 8, 0.025, 8, false
    ),
    railMat
  )
  scene.add(innerRail)

  // ── BALUSTERS (every 3 steps) ────────────────────────────────
  for (let i = 0; i < STEP_COUNT; i += 3) {
    const a = i * APR
    const y = START_Y - (i * STEP_H)
    const outerR = STAND_R + STEP_W / 2 - 0.05
    const bal = new THREE.Mesh(
      new THREE.CylinderGeometry(0.02, 0.02, 0.85, 6),
      railMat
    )
    bal.position.set(
      Math.cos(a) * outerR,
      y + 0.425,
      Math.sin(a) * outerR
    )
    scene.add(bal)
  }

  // ── TOP OF SHAFT: decorative cap ─────────────────────────────
  const capRingGeom = new THREE.TorusGeometry(SHAFT_R + 0.15, 0.12, 8, 32)
  const capMat = new THREE.MeshStandardMaterial({
    color: 0x333333, metalness: 0.5, roughness: 0.5
  })
  const capRing = new THREE.Mesh(capRingGeom, capMat)
  capRing.rotation.x = Math.PI / 2
  capRing.position.y = 8.5
  scene.add(capRing)

  const capRing2 = capRing.clone()
  capRing2.position.y = 7.8
  scene.add(capRing2)

  // Shaft cap - solid cylinder
  const shaftCap = new THREE.Mesh(
    new THREE.CylinderGeometry(SHAFT_R + 0.3, SHAFT_R + 0.1, 0.3, 32),
    new THREE.MeshStandardMaterial({ color: 0x1a1a1a, roughness: 0.8, metalness: 0.2 })
  )
  shaftCap.position.y = 8.65
  scene.add(shaftCap)

  // ── MOCK STAIRS ABOVE (continuing upward into darkness) ──────
  const mockAboveCount = 20
  const mockTreadMat = new THREE.MeshStandardMaterial({
    color: 0x1e1e1e, roughness: 0.95, metalness: 0.0,
    transparent: true, opacity: 0.6
  })
  const mockRiserMat = new THREE.MeshStandardMaterial({
    color: 0x161616, roughness: 0.95, metalness: 0.0,
    transparent: true, opacity: 0.6
  })

  for (let i = 0; i < mockAboveCount; i++) {
    const angle = (i + 1) * APR  // continue from top step
    const y = START_Y + (i + 1) * STEP_H
    const cx = Math.cos(angle) * STAND_R
    const cz = Math.sin(angle) * STAND_R

    const tread = new THREE.Mesh(
      new THREE.BoxGeometry(STEP_W, 0.045, STEP_D),
      mockTreadMat
    )
    tread.castShadow = true
    tread.receiveShadow = true

    const riser = new THREE.Mesh(
      new THREE.BoxGeometry(STEP_W, STEP_H, 0.04),
      mockRiserMat
    )
    riser.position.set(0, -STEP_H / 2, STEP_D / 2 - 0.02)

    const sg = new THREE.Group()
    sg.add(tread)
    sg.add(riser)
    sg.position.set(cx, y, cz)
    sg.rotation.y = -angle + Math.PI / 2
    scene.add(sg)
  }

  // ── MOCK STAIRS BELOW (continuing downward into darkness) ────
  const mockBelowCount = 20

  for (let i = 0; i < mockBelowCount; i++) {
    const angle = (STEP_COUNT + i) * APR
    const y = START_Y - (STEP_COUNT + i) * STEP_H
    const cx = Math.cos(angle) * STAND_R
    const cz = Math.sin(angle) * STAND_R

    const tread = new THREE.Mesh(
      new THREE.BoxGeometry(STEP_W, 0.045, STEP_D),
      mockTreadMat
    )
    tread.castShadow = true
    tread.receiveShadow = true

    const riser = new THREE.Mesh(
      new THREE.BoxGeometry(STEP_W, STEP_H, 0.04),
      mockRiserMat
    )
    riser.position.set(0, -STEP_H / 2, STEP_D / 2 - 0.02)

    const sg = new THREE.Group()
    sg.add(tread)
    sg.add(riser)
    sg.position.set(cx, y, cz)
    sg.rotation.y = -angle + Math.PI / 2
    scene.add(sg)
  }

  // ── MOCK RAILINGS ABOVE ──────────────────────────────────────
  const mockRailMat = new THREE.MeshStandardMaterial({
    color: 0x3a3a3a, roughness: 0.4, metalness: 0.7,
    transparent: true, opacity: 0.4
  })

  const mockOuterPts = []
  for (let i = 0; i <= mockAboveCount + 5; i++) {
    const a = (i + STEP_COUNT) * APR
    const r = STAND_R + STEP_W / 2 - 0.05
    mockOuterPts.push(new THREE.Vector3(
      Math.cos(a) * r,
      START_Y - (STEP_COUNT + i) * STEP_H + 0.85,
      Math.sin(a) * r
    ))
  }
  scene.add(new THREE.Mesh(
    new THREE.TubeGeometry(
      new THREE.CatmullRomCurve3(mockOuterPts),
      mockAboveCount * 4, 0.025, 6, false
    ),
    mockRailMat
  ))

  // ── VERIFY ───────────────────────────────────────────────────
  console.log('Sample stair positions:')
  stairPositions.slice(0, 3).forEach((p, i) => {
    const d = Math.sqrt(p.x ** 2 + p.z ** 2)
    console.log(`Step ${i}: radius=${d.toFixed(2)} (shaft=${SHAFT_R})`,
      d > SHAFT_R + 0.5 ? '✓ OUTSIDE' : '✗ TOO CLOSE')
  })

  return { stairPositions, shaft }
}
