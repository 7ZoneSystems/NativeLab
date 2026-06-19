import * as THREE from 'three'
import gsap from 'gsap'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

export function createAvatar(onProgress) {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader()

    loader.load('models/avatar.glb', (gltf) => {
      const avatar = gltf.scene

      const boxRaw = new THREE.Box3().setFromObject(avatar)
      const rawHeight = boxRaw.max.y - boxRaw.min.y
      const targetHeight = 1.75
      const scaleFactor = targetHeight / rawHeight
      avatar.scale.setScalar(scaleFactor)

      const boxScaled = new THREE.Box3().setFromObject(avatar)
      const feetOffset = boxScaled.min.y

      // Load NativeLab logo
      const logoTex = new THREE.TextureLoader().load('nativelab-icon.png')
      logoTex.colorSpace = THREE.SRGBColorSpace

      // Metallic green-violet suit - dark, reflective, picks up env map
      const suitGreen = 0x1a5a3a
      const suitViolet = 0x3a1e5a

      avatar.traverse((child) => {
        if (!child.isMesh) return
        const n = child.name.toLowerCase()

        // Default: dark metallic green - reflective
        child.material = new THREE.MeshStandardMaterial({
          color: suitGreen,
          roughness: 0.15,
          metalness: 0.85,
          envMapIntensity: 1.2,
          emissive: 0x030a05,
          emissiveIntensity: 0.03
        })

        // Chest/torso - dark metallic violet - most reflective
        if (n.includes('chest') || n.includes('torso') ||
            n.includes('body') || n.includes('upper')) {
          child.material = new THREE.MeshStandardMaterial({
            color: suitViolet,
            roughness: 0.1,
            metalness: 0.9,
            envMapIntensity: 1.5,
            emissive: 0x060310,
            emissiveIntensity: 0.04
          })
        }

        // Hands - dark green, reflective
        if (n.includes('hand') || n.includes('glove') ||
            n.includes('finger')) {
          child.material = new THREE.MeshStandardMaterial({
            color: 0x123a28,
            roughness: 0.2,
            metalness: 0.75,
            envMapIntensity: 1.0,
            emissive: 0x030805,
            emissiveIntensity: 0.02
          })
        }

        // Legs - dark violet, reflective
        if (n.includes('leg') || n.includes('thigh') ||
            n.includes('shin') || n.includes('knee') ||
            n.includes('lower')) {
          child.material = new THREE.MeshStandardMaterial({
            color: 0x2a1848,
            roughness: 0.12,
            metalness: 0.82,
            envMapIntensity: 1.2,
            emissive: 0x040210,
            emissiveIntensity: 0.02
          })
        }

        // Dark visor - mirror-like
        if (n.includes('head') || n.includes('helmet') ||
            n.includes('visor') || n.includes('face')) {
          child.material = new THREE.MeshStandardMaterial({
            color: 0x080800,
            roughness: 0.0,
            metalness: 1.0,
            envMapIntensity: 2.0
          })
        }

        // Backpack / boots - dark metallic
        if (n.includes('pack') || n.includes('detail') ||
            n.includes('boot') || n.includes('shoe') ||
            n.includes('foot')) {
          child.material = new THREE.MeshStandardMaterial({
            color: 0x1a1a25,
            roughness: 0.3,
            metalness: 0.7,
            envMapIntensity: 0.8,
            emissive: 0x000000,
            emissiveIntensity: 0
          })
        }

        child.castShadow = true
        child.receiveShadow = true
      })

      // ── NativeLab logo - attached to avatar root, on chest ───
      // Find approximate chest position by checking bounding box
      const chestY = feetOffset + (boxScaled.max.y - feetOffset) * 0.65
      const logoPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(0.4, 0.2),
        new THREE.MeshBasicMaterial({
          map: logoTex,
          transparent: true,
          side: THREE.DoubleSide,
          depthTest: true
        })
      )
      logoPlane.name = 'nativelabLogo'
      logoPlane.position.set(0, chestY, 0.2)
      avatar.add(logoPlane)
      console.log('Logo placed at Y:', chestY.toFixed(2), 'Z: 0.2 (on chest)')

      // ── Green patch on shoulder ────────────────────────────────
      const patchMat = new THREE.MeshStandardMaterial({
        color: 0x46be3c, side: THREE.DoubleSide,
        emissive: 0x46be3c, emissiveIntensity: 0
      })

      const shoulder = avatar.getObjectByName('mixamorigRightShoulder')
                    || avatar.getObjectByName('RightShoulder')
                    || avatar.getObjectByName('Shoulder_R')
                    || avatar.getObjectByName('RightArm')

      if (shoulder) {
        const patch = new THREE.Mesh(new THREE.CircleGeometry(0.06, 16), patchMat)
        patch.name = 'patch'
        patch.position.set(0, 0, 0.08)
        shoulder.add(patch)
      }

      // ── AnimationMixer ───────────────────────────────────────
      const mixer = new THREE.AnimationMixer(avatar)
      const clips = gltf.animations

      let idleAction = null
      let walkAction = null
      let currentAction = null

      const idleClip = THREE.AnimationClip.findByName(clips, 'Idle')
                     || THREE.AnimationClip.findByName(clips, 'idle')
                     || clips[0]

      const walkClip = THREE.AnimationClip.findByName(clips, 'Walk')
                    || THREE.AnimationClip.findByName(clips, 'walk')
                    || THREE.AnimationClip.findByName(clips, 'Running')

      if (idleClip) {
        idleAction = mixer.clipAction(idleClip)
        idleAction.play()
        currentAction = idleAction
      }
      if (walkClip) {
        walkAction = mixer.clipAction(walkClip)
      }

      function playIdle() {
        if (!idleAction || currentAction === idleAction) return
        idleAction.reset().play()
        if (currentAction) idleAction.crossFadeFrom(currentAction, 0.3, true)
        currentAction = idleAction
      }

      function playWalk() {
        if (!walkAction || currentAction === walkAction) return
        walkAction.reset().play()
        if (currentAction) walkAction.crossFadeFrom(currentAction, 0.2, true)
        currentAction = walkAction
      }

      function glowPatch() {
        gsap.to(patchMat, { emissiveIntensity: 0.6, duration: 0.3, yoyo: true, repeat: 3 })
      }

      resolve({
        avatar, mixer, playIdle, playWalk, glowPatch,
        feetOffset, avatarHeight: targetHeight
      })
    }, onProgress, reject)
  })
}
