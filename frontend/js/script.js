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

document.addEventListener("DOMContentLoaded", () => {
  const ctx = document.getElementById("priceChart").getContext("2d");

  const chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Gold price",
        data: [],
        borderColor: "red",
        backgroundColor: "rgba(255,0,0,0.1)",
        tension: 0.2,
        fill: false,
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: {
          ticks: {
            maxRotation: 45,
            minRotation: 30
          }
        },
        y: {
          beginAtZero: false
        }
      }
    }
  });

  const periodToInterval = {
    hour:  "1h",
    day:   "90m",
    week:  "1d",
    month: "1wk",
    year:  "1mo"
  };

  const periodToPeriodCode = {
    hour:  "1d",
    day:   "5d",
    week:  "1mo",
    month: "6mo",
    year:  "1y"
  };

  async function fetchData(periodKey) {
    const period = periodToPeriodCode[periodKey];
    const interval = periodToInterval[periodKey];
    const response = await fetch(`http://localhost:8000/metals/historical_data?metal_id=Gold&period=${period}&interval=${interval}`);
    
    if (!response.ok) {
      console.error("Error", response.status);
      return;
    }

    const data = await response.json();

    const labels = data.map(point => {
      const date = new Date(point.timestamp);
      if (isNaN(date)) {
        console.warn("Invalid timestamp:", point.timestamp);
        return "";
      }
      return date.toLocaleString();
    });

    const prices = data.map(point => point.close);

    chart.data.labels = labels;
    chart.data.datasets[0].data = prices;
    chart.update();
  }

  document.querySelectorAll(".graph__button").forEach(button => {
    button.addEventListener("click", () => {
      const periodKey = button.dataset.interval;
      fetchData(periodKey);
    });
  });

  fetchData("day");
});