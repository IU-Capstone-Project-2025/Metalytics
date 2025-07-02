let selectedDate = 'day';

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
  let currentPeriod = "day";
  let currentSelect = "historical";
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
    hour:  "1m",
    day:   "1h",
    week:  "1d",
    month: "1wk",
    year:  "1mo"
  };

  const periodToPeriodCode = {
    hour:  "1h",
    day:   "1d",
    week:  "1wk",
    month: "6mo",
    year:  "1y"
  };

  const periodToForecast = {
    hour:  { unit: "m", value: 60 },
    day:   { unit: "h", value: 24 },
    week:  { unit: "d", value: 7 },
    month: { unit: "d", value: 30 },
    year:  { unit: "d", value: 365 }
  };

  function setSelectButtonActive(selected) {
    document.querySelectorAll('[data-select]').forEach(btn => {
      if (btn.dataset.select === selected) {
        btn.classList.add('active');
      } 
      
      else {
        btn.classList.remove('active');
      }
    });
  }

  async function fetchData(periodKey) {
    const period = periodToPeriodCode[periodKey];
    const interval = periodToInterval[periodKey];
    const response = await fetch(`http://localhost:8000/historical_data/Gold?period=${period}&interval=${interval}`);
    
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

  async function fetchPredictedData(periodKey) {
    const forecastParams = periodToForecast[periodKey] || { unit: "h", value: 24 };
    const response = await fetch(`http://localhost:8000/forecast/Gold/days?unit=${forecastParams.unit}&value=${forecastParams.value}`);
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
    const prices = data.map(point => point.price);
    chart.data.labels = labels;
    chart.data.datasets[0].data = prices;
    chart.data.datasets[0].label = "Predicted Gold price";
    chart.data.datasets[0].borderColor = "blue";
    chart.data.datasets[0].backgroundColor = "rgba(0,0,255,0.1)";
    chart.update();
  }

  function activeSelect() {
    document.querySelectorAll('[data-select]').forEach(button => {
      if (button.dataset.select === currentSelect) {
        button.classList.add('graph__button--active');
      } 
      
      else {
        button.classList.remove('graph__button--active');
      }
    });
  }

  document.querySelectorAll(".graph__button").forEach(button => {
    button.addEventListener("click", () => {
      if (button.dataset.interval) {
        currentPeriod = button.dataset.interval;
        if (currentSelect === "historical") {
          chart.data.datasets[0].borderColor = "red";
          chart.data.datasets[0].backgroundColor = "rgba(255,0,0,0.1)";
          chart.data.datasets[0].label = "Gold price";
          fetchData(currentPeriod);
        } else if (currentSelect === "predicted") {
          fetchPredictedData(currentPeriod);
        }
      }
      if (button.dataset.select) {
        currentSelect = button.dataset.select;
        setSelectButtonActive(currentSelect);
        activeSelect();
        if (currentSelect === "historical") {
          chart.data.datasets[0].borderColor = "red";
          chart.data.datasets[0].backgroundColor = "rgba(255,0,0,0.1)";
          chart.data.datasets[0].label = "Gold price";
          fetchData(currentPeriod);
        } else if (currentSelect === "predicted") {
          fetchPredictedData(currentPeriod);
        }
      }
    });
  });

  fetchData("day");

  document.querySelectorAll('[data-metal]').forEach(button => {
    button.addEventListener('click', () => {
      selectedMetal = button.dataset.metal;
      activeMetal();
    });
  });

  document.querySelectorAll('[data-interval]').forEach(button => {
    button.addEventListener('click', () => {
      selectedDate = button.dataset.interval;
      activeInterval();
    });
  });

  activeInterval();
  activeSelect();

  addRippleEffectToMetalButtons();
});

function activeInterval() {
  document.querySelectorAll('[data-interval]').forEach(button => {
    if (button.dataset.interval === selectedDate) {
      button.classList.add('graph__button--active');
    } 
    
    else {
      button.classList.remove('graph__button--active');
    }
  });
  
}

function activeMetal() {
  document.querySelectorAll('[data-metal]').forEach(button => {
    if (button.dataset.metal === selectedMetal) {
      button.classList.add('header__button--active');
    } 
    
    else {
      button.classList.remove('header__button--active');
    }
  });
}

function addRippleEffectToMetalButtons() {
  document.querySelectorAll('.header__button').forEach(button => {
    button.addEventListener('click', function(e) {
      const oldRipple = button.querySelector('.ripple');
      if (oldRipple) oldRipple.remove();
      const rect = button.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;

      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      ripple.style.width = ripple.style.height = size + 'px';
      ripple.style.left = x + 'px';
      ripple.style.top = y + 'px';

      const borderColor = getComputedStyle(button).getPropertyValue('--button');
      ripple.style.backgroundColor = borderColor.trim();
      button.appendChild(ripple);

      ripple.addEventListener('animationend', () => ripple.remove());
    });
  });
}
