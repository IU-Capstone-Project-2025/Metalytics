window.addEventListener("DOMContentLoaded", () => {
  fetch("http://localhost:8000/")
    .then(response => response.json())
    .then(data => {
      document.getElementById("response").textContent = data.message;
    })
    .catch(error => {
      document.getElementById("response").textContent = "Error: " + error;
    });
});

const swiper = new Swiper('.swiper', {
    direction: 'horizontal',
    loop: true,
    effect: 'creative',
    creativeEffect: {
        prev: {
        shadow: true,
        translate: ['-120%', 0, -500],
        },
        next: {
        shadow: true,
        translate: ['120%', 0, -500],
        },
    },
    speed: 600,
    spaceBetween: 100,

    pagination: {
        el: '.swiper-pagination',
        clickable: true
    },

    navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev',
    },
});