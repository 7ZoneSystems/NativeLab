import * as THREE from 'three'

const SHAFT_R = 1.8
const STAND_R = 2.85
const STEPS = 40
const ANGLE_PER_STEP = (Math.PI * 2 * 3.0) / STEPS

export class CameraFollow {
  constructor(camera) {
    this.camera = camera
    this.smoothPos = new THREE.Vector3(0, 10, 16)
    this.smoothLook = new THREE.Vector3(0, 5, 0)
    this.lerpSpeed = 2.5
    this._time = 0
  }

  update(delta, avatarPosition, scrollProgress) {
    this._time += delta
    let targetPos, targetLook

    if (scrollProgress < 0.12) {
      // FULL DARK - camera far back, looking at nothing
      targetPos = new THREE.Vector3(0, 10, 16)
      targetLook = new THREE.Vector3(0, 5, 0)

    } else if (scrollProgress < 0.22) {
      // "Hi I'm Hrirake" - camera slowly pushes toward robot
      const t = (scrollProgress - 0.12) / 0.1
      targetPos = new THREE.Vector3(
        avatarPosition.x + 1.5,
        avatarPosition.y + 2.0,
        avatarPosition.z + 5.0
      )
      targetLook = avatarPosition.clone().add(new THREE.Vector3(0, 1.0, 0))

    } else if (scrollProgress < 0.82) {
      // FOLLOW - camera always in FRONT of robot on spiral
      // Calculate angle from avatar to center
      const avatarAngle = Math.atan2(avatarPosition.z, avatarPosition.x)
      // Camera goes to OPPOSITE side of shaft from avatar
      // This ensures shaft is NOT between camera and robot
      const camAngle = avatarAngle + Math.PI  // opposite side
      const camRadius = STAND_R + 4.0  // well outside everything

      // But we want camera BEHIND avatar on spiral, not opposite
      // Behind = same side as avatar, but further out
      const behindAngle = avatarAngle - 0.5

      targetPos = new THREE.Vector3(
        Math.cos(behindAngle) * camRadius,
        avatarPosition.y + 2.5,
        Math.sin(behindAngle) * camRadius
      )

      // Look at robot chest/head
      targetLook = avatarPosition.clone().add(new THREE.Vector3(0, 1.2, 0))

      // Subtle handheld shake
      targetPos.x += Math.sin(this._time * 7) * 0.025
      targetPos.y += Math.cos(this._time * 5) * 0.015

    } else if (scrollProgress < 0.92) {
      // CLOSE UP - face level, offset to side so shaft not blocking
      const avatarAngle = Math.atan2(avatarPosition.z, avatarPosition.x)
      const camAngle = avatarAngle - 0.3

      targetPos = new THREE.Vector3(
        Math.cos(camAngle) * 2.5 + avatarPosition.x,
        avatarPosition.y + 1.6,
        Math.sin(camAngle) * 2.5 + avatarPosition.z
      )
      targetLook = new THREE.Vector3(
        avatarPosition.x,
        avatarPosition.y + 1.4,
        avatarPosition.z
      )

    } else {
      // LOGO FOCUS - chest level, close, offset slightly right
      const avatarAngle = Math.atan2(avatarPosition.z, avatarPosition.x)
      const camAngle = avatarAngle + 0.1

      targetPos = new THREE.Vector3(
        Math.cos(camAngle) * 1.6 + avatarPosition.x,
        avatarPosition.y + 0.85,
        Math.sin(camAngle) * 1.6 + avatarPosition.z
      )
      targetLook = new THREE.Vector3(
        avatarPosition.x,
        avatarPosition.y + 0.75,
        avatarPosition.z
      )
    }

    // ── CLAMP: camera never inside shaft ───────────────────────
    const camDist = Math.sqrt(targetPos.x * targetPos.x + targetPos.z * targetPos.z)
    if (camDist < SHAFT_R + 0.8) {
      const dir = new THREE.Vector3(targetPos.x, 0, targetPos.z).normalize()
      targetPos.x = dir.x * (SHAFT_R + 0.8)
      targetPos.z = dir.z * (SHAFT_R + 0.8)
    }

    // ── CLAMP: minimum distance from avatar ────────────────────
    const distToAvatar = targetPos.distanceTo(avatarPosition)
    if (distToAvatar < 2.5) {
      const dir = targetPos.clone().sub(avatarPosition).normalize()
      targetPos.copy(avatarPosition).add(dir.multiplyScalar(2.5))
    }

    // Smooth lerp
    this.smoothPos.lerp(targetPos, delta * this.lerpSpeed)
    this.smoothLook.lerp(targetLook, delta * this.lerpSpeed)

    this.camera.position.copy(this.smoothPos)
    this.camera.lookAt(this.smoothLook)
  }
}
