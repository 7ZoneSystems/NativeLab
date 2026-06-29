import { defineConfig } from 'vite'
import { resolve } from 'path'
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from 'fs'

function copyStaticFiles() {
  return {
    name: 'copy-static-files',
    closeBundle() {
      const srcDir = resolve(__dirname)
      const distDir = resolve(__dirname, 'dist')

      const filesToCopy = [
        'features.html',
        'phonolab.html',
        'pipeline.html',
        'docs-user.html',
        'compare.html',
        'setup.html',
        'docs-dev.html',
        'site.css',
        'site.js',
        'Phonolabv0.9.apk'
      ]

      for (const file of filesToCopy) {
        const src = resolve(srcDir, file)
        const dest = resolve(distDir, file)
        if (existsSync(src)) {
          copyFileSync(src, dest)
        }
      }

      const publicDir = resolve(srcDir, 'public')
      if (existsSync(publicDir)) {
        function copyRecursive(src, dest) {
          if (statSync(src).isDirectory()) {
            if (!existsSync(dest)) mkdirSync(dest, { recursive: true })
            for (const file of readdirSync(src)) {
              copyRecursive(resolve(src, file), resolve(dest, file))
            }
          } else {
            copyFileSync(src, dest)
          }
        }
        copyRecursive(publicDir, distDir)
      }
    }
  }
}

export default defineConfig({
  // './' for local dev, '/NativeLab/' for GitHub Pages production build
  base: process.env.NODE_ENV === 'production' ? '/NativeLab/' : './',
  server: {
    fs: {
      allow: ['..']
    }
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        features: resolve(__dirname, 'features.html'),
        phonolab: resolve(__dirname, 'phonolab.html'),
        pipeline: resolve(__dirname, 'pipeline.html'),
        'docs-user': resolve(__dirname, 'docs-user.html'),
        compare: resolve(__dirname, 'compare.html'),
        setup: resolve(__dirname, 'setup.html'),
        'docs-dev': resolve(__dirname, 'docs-dev.html'),
      }
    }
  },
  plugins: [copyStaticFiles()]
})