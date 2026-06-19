// NativeLab - Starfield + 3D Helix Scroll Engine

(function () {
  'use strict';

  // ════════════════════════════════════════════════════════════
  // STARFIELD - Canvas background with depth
  // ════════════════════════════════════════════════════════════

  class Starfield {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.stars = [];
      this.scrollY = 0;
      this.resize();
      this.init();
      this.animate();

      window.addEventListener('resize', () => this.resize());
      window.addEventListener('scroll', () => {
        this.scrollY = window.scrollY;
      }, { passive: true });
    }

    resize() {
      this.canvas.width = window.innerWidth;
      this.canvas.height = window.innerHeight;
    }

    init() {
      this.stars = [];
      const count = Math.min(200, Math.floor(window.innerWidth * window.innerHeight / 5000));
      for (let i = 0; i < count; i++) {
        this.stars.push({
          x: Math.random() * this.canvas.width,
          y: Math.random() * this.canvas.height,
          z: Math.random() * 3 + 0.5,
          size: Math.random() * 1.5 + 0.3,
          brightness: Math.random() * 0.6 + 0.2,
          twinkle: Math.random() * Math.PI * 2,
          twinkleSpeed: Math.random() * 0.03 + 0.005,
          isGreen: Math.random() < 0.15,
        });
      }
    }

    animate() {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

      // Draw stars with parallax based on scroll
      for (const star of this.stars) {
        // Parallax: deeper stars move slower
        const parallaxY = (this.scrollY * 0.05 * star.z) % this.canvas.height;
        let drawY = (star.y - parallaxY + this.canvas.height) % this.canvas.height;

        // Twinkle
        star.twinkle += star.twinkleSpeed;
        const twinkleAlpha = star.brightness + Math.sin(star.twinkle) * 0.2;
        const alpha = Math.max(0.05, Math.min(1, twinkleAlpha));

        // Size based on depth
        const drawSize = star.size * (star.z * 0.4);

        // Color
        if (star.isGreen) {
          this.ctx.fillStyle = `rgba(0, 255, 136, ${alpha})`;
        } else {
          this.ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.7})`;
        }

        this.ctx.beginPath();
        this.ctx.arc(star.x, drawY, drawSize, 0, Math.PI * 2);
        this.ctx.fill();

        // Subtle glow for green stars
        if (star.isGreen && alpha > 0.3) {
          this.ctx.fillStyle = `rgba(0, 255, 136, ${alpha * 0.15})`;
          this.ctx.beginPath();
          this.ctx.arc(star.x, drawY, drawSize * 3, 0, Math.PI * 2);
          this.ctx.fill();
        }
      }

      requestAnimationFrame(() => this.animate());
    }
  }

  // ════════════════════════════════════════════════════════════
  // 3D HELIX SCROLL - Content appears along a spiral path
  // ════════════════════════════════════════════════════════════

  class HelixScroll {
    constructor() {
      this.items = [];
      this.scrollY = 0;
      this.viewportH = window.innerHeight;
      this.init();

      window.addEventListener('scroll', () => {
        this.scrollY = window.scrollY;
        this.update();
      }, { passive: true });

      window.addEventListener('resize', () => {
        this.viewportH = window.innerHeight;
        this.update();
      });
    }

    init() {
      // Find all helix items
      document.querySelectorAll('[data-helix]').forEach((el, index) => {
        const side = el.getAttribute('data-helix') || (index % 2 === 0 ? 'left' : 'right');
        const depth = parseFloat(el.getAttribute('data-depth') || '0');
        this.items.push({ el, side, depth, index, lastProgress: -1 });
      });

      this.update();
    }

    update() {
      for (const item of this.items) {
        const rect = item.el.getBoundingClientRect();
        const elTop = rect.top;
        const elHeight = rect.height;

        // Calculate how far the element is through the viewport
        // 0 = just entering bottom, 1 = just leaving top
        const progress = 1 - (elTop / this.viewportH);

        // Only animate when element is near viewport
        if (progress < -0.2 || progress > 1.3) continue;

        // Clamp progress for animation range
        const animProgress = Math.max(0, Math.min(1, (progress + 0.1) / 0.8));

        // Helix path: element comes from the side, moves to center, then continues
        const side = item.side;
        const depth = item.depth;

        // Horizontal movement: starts off-screen, moves to center
        const startX = side === 'left' ? -400 : 400;
        const translateX = startX * (1 - animProgress);

        // Vertical offset for helix effect (slight sine wave)
        const helixOffset = Math.sin(animProgress * Math.PI) * 30;

        // Depth: comes from far away, settles at 0
        const translateZ = depth * (1 - animProgress);

        // Scale: starts small, grows to full
        const scale = 0.7 + (animProgress * 0.3);

        // Opacity: fades in
        const opacity = Math.min(1, animProgress * 2);

        // Apply transform (no rotation, just position + depth + scale)
        item.el.style.transform = `perspective(1200px) translateX(${translateX}px) translateY(${helixOffset}px) translateZ(${translateZ}px) scale(${scale})`;
        item.el.style.opacity = opacity;
        item.el.style.transition = 'none'; // Direct scroll control, no CSS transition
      }
    }
  }

  // ════════════════════════════════════════════════════════════
  // COUNTER ANIMATION
  // ════════════════════════════════════════════════════════════

  class CounterEngine {
    constructor() {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const el = entry.target;
            const target = parseInt(el.getAttribute('data-count'), 10);
            this.animate(el, target, 1500);
            observer.unobserve(el);
          }
        });
      }, { threshold: 0.5 });

      document.querySelectorAll('[data-count]').forEach(el => observer.observe(el));
    }

    animate(el, target, duration) {
      const startTime = performance.now();
      const update = (currentTime) => {
        const progress = Math.min((currentTime - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(update);
      };
      requestAnimationFrame(update);
    }
  }

  // ════════════════════════════════════════════════════════════
  // 3D CARD TILT - Mouse-driven, not scroll
  // ════════════════════════════════════════════════════════════

  class CardTilt {
    constructor() {
      document.querySelectorAll('.card-3d').forEach(card => {
        const inner = card.querySelector('.card-3d-inner') || card;
        card.addEventListener('mousemove', (e) => {
          const rect = card.getBoundingClientRect();
          const x = e.clientX - rect.left;
          const y = e.clientY - rect.top;
          const centerX = rect.width / 2;
          const centerY = rect.height / 2;
          const rotateX = ((y - centerY) / centerY) * -8;
          const rotateY = ((x - centerX) / centerX) * 8;
          inner.style.transform = `perspective(1200px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateZ(10px)`;
        });
        card.addEventListener('mouseleave', () => {
          inner.style.transform = 'perspective(1200px) rotateX(0) rotateY(0) translateZ(0)';
        });
      });
    }
  }

  // ════════════════════════════════════════════════════════════
  // FADE-UP ANIMATION - IntersectionObserver based
  // ════════════════════════════════════════════════════════════

  class FadeUp {
    constructor() {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('in-view');
            observer.unobserve(entry.target);
          }
        });
      }, { threshold: 0.15 });

      document.querySelectorAll('.fade-up').forEach(el => observer.observe(el));
    }
  }

  // ════════════════════════════════════════════════════════════
  // INIT
  // ════════════════════════════════════════════════════════════

  // Starfield
  const canvas = document.getElementById('starfield');
  if (canvas) new Starfield(canvas);

  // 3D Helix scroll
  new HelixScroll();

  // Counters
  new CounterEngine();

  // Card tilt (mouse only)
  new CardTilt();

  // Fade-up animations
  new FadeUp();

  // Smooth scroll for in-page anchors
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', (e) => {
      const href = a.getAttribute('href');
      if (href === '#' || href.length < 2) return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        window.scrollTo({ top: target.offsetTop - 80, behavior: 'smooth' });
      }
    });
  });

  // Active nav link
  const here = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
  document.querySelectorAll('.nav-links a').forEach(a => {
    const href = (a.getAttribute('href') || '').toLowerCase();
    if (!href || href.startsWith('http') || href.startsWith('#')) return;
    if (href === here || (here === '' && href === 'index.html')) {
      a.classList.add('active');
    }
  });

})();
