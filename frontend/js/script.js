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
});
