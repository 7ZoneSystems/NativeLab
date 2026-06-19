export const isMobile = window.innerWidth < 768

export function getMobileConfig() {
  if (!isMobile) return null

  return {
    pixelRatio: 1,
    bloomStrength: 0.2,
    bloomEnabled: false,
    filmEnabled: false,
    smaaEnabled: false,
    stairCount: 16,
    shadowMapSize: 512,
    fpsCap: 30,
    simplifiedAvatar: true,
    cameraFov: 60,
    featureTagsProjected: false
  }
}
