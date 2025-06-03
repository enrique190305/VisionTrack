// src/static/js/main.js

document.addEventListener('DOMContentLoaded', function() {
    console.log("La aplicación web está lista.");

    // Animación de números en las estadísticas
    const statNumbers = document.querySelectorAll('.stat-number');
    
    const animateValue = (element, start, end, duration) => {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const value = Math.floor(progress * (end - start) + start);
            element.textContent = value + (element.dataset.suffix || '');
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    };

    // Iniciar animaciones cuando los elementos son visibles
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const el = entry.target;
                const value = parseInt(el.textContent);
                animateValue(el, 0, value, 2000);
                observer.unobserve(el);
            }
        });
    }, { threshold: 0.5 });

    statNumbers.forEach(stat => observer.observe(stat));

    // Menú móvil
    const mobileMenuBtn = document.querySelector('.mobile-menu');
    const mainNav = document.querySelector('.main-nav');

    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', () => {
            mainNav.classList.toggle('active');
        });
    }

    // Cerrar menú al hacer click fuera
    document.addEventListener('click', function(e) {
        if (!mainNav.contains(e.target) && !mobileMenuBtn.contains(e.target)) {
            mainNav.classList.remove('active');
        }
    });

    // Cerrar menú al redimensionar la ventana
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768) {
            mainNav.classList.remove('active');
        }
    });

    // Efecto de scroll suave
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Animación al hacer scroll
    const fadeInElements = document.querySelectorAll('.feature-card, .stat-card');
    
    const fadeInObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    fadeInElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
        fadeInObserver.observe(el);
    });
});
