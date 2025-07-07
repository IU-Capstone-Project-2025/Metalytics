let selectedDate = 'day';
let selectedMetal = 'gold';

let metalNames = {
  gold: 'GOLD',
  silver: 'SILVER',
  zinc: 'ZINC'
};

const metalNews = {
  gold: [
    {
      info: "Gold prices reached new highs today as investors seek safe haven assets amid market volatility. The precious metal continues to show strong performance in uncertain economic conditions.",
      date: "10.00, 24.04.2025"
    },
    {
      info: "Central banks around the world continue to increase their gold reserves, signaling confidence in the metal's long-term value. This trend is expected to support gold prices in the coming months.",
      date: "09.30, 24.04.2025"
    },
    {
      info: "Gold mining companies report increased production costs due to rising energy prices, which may impact supply and contribute to price stability in the gold market.",
      date: "09.00, 24.04.2025"
    },
    {
      info: "Technical analysis suggests gold is forming a bullish pattern, with key resistance levels being tested. Traders are watching for potential breakout opportunities.",
      date: "08.30, 24.04.2025"
    }
  ],
  silver: [
    {
      info: "silver ore prices show mixed signals as Chinese steel production remains strong but global demand concerns persist. Market analysts are closely monitoring supply chain developments.",
      date: "10.00, 24.04.2025"
    },
    {
      info: "Major silver ore producers announce production cuts in response to market conditions, which could tighten supply and support prices in the short term.",
      date: "09.30, 24.04.2025"
    },
    {
      info: "Steel industry demand for silver ore remains robust in Asia, while European markets show signs of recovery. This regional divergence is creating interesting trading opportunities.",
      date: "09.00, 24.04.2025"
    },
    {
      info: "Envsilvermental regulations are impacting silver ore mining operations, leading to increased production costs and potential supply constraints in certain regions.",
      date: "08.30, 24.04.2025"
    }
  ],
  zinc: [
    {
      info: "Zinc prices are climbing as supply concerns mount due to mine closures and production disruptions. The metal's essential role in galvanization continues to drive demand.",
      date: "10.00, 24.04.2025"
    },
    {
      info: "Electric vehicle battery demand is creating new opportunities for zinc producers, as the metal is increasingly used in advanced battery technologies and energy storage solutions.",
      date: "09.30, 24.04.2025"
    },
    {
      info: "Zinc inventories at major exchanges are declining, indicating strong physical demand and potential for further price increases in the coming weeks.",
      date: "09.00, 24.04.2025"
    },
    {
      info: "Construction sector demand for zinc remains strong, particularly in emerging markets where infrastructure development is accelerating rapidly.",
      date: "08.30, 24.04.2025"
    }
  ]
};

let swiper;

document.addEventListener("DOMContentLoaded", () => {
  swiper = new Swiper('.swiper', {
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

  let currentPeriod = "day";
  let currentSelect = "historical";
  const ctx = document.getElementById("priceChart").getContext("2d");

  function getLegendDisplay() {
    return window.innerWidth > 768;
  }

  function getXAxisOptions() {
    if (window.innerWidth <= 768) {
      return {
        ticks: {
          display: false
        }
      };
    } else {
      return {
        ticks: {
          maxRotation: 45,
          minRotation: 30,
          callback: function(value, index, values) {
            const label = this.getLabelForValue(value);
            let dateObj = new Date(label);
            if (isNaN(dateObj)) return label;
            let day = String(dateObj.getDate()).padStart(2, '0');
            let month = String(dateObj.getMonth() + 1).padStart(2, '0');
            let year = String(dateObj.getFullYear()).slice(-2);
            let hours = String(dateObj.getHours()).padStart(2, '0');
            let minutes = String(dateObj.getMinutes()).padStart(2, '0');
            return `${day}.${month}.${year}, ${hours}:${minutes}`;
          }
        }
      };
    }
  }

  function shortNumberFormat(num) {
    if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(1) + 'M';
    if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(1);
  }

  function getYAxisOptions() {
    if (window.innerWidth <= 768) {
      return {
        ticks: {
          display: false
        }
      };
    } else {
      return {
        ticks: {
          callback: function(value) {
            return shortNumberFormat(value);
          }
        }
      };
    }
  }

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
      plugins: {
        legend: {
          display: getLegendDisplay()
        }
      },
      scales: {
        x: getXAxisOptions(),
        y: getYAxisOptions()
      }
    }
  });

  window.addEventListener('resize', function() {
    const shouldShowLegend = getLegendDisplay();
    if (chart.options.plugins.legend.display !== shouldShowLegend) {
      chart.options.plugins.legend.display = shouldShowLegend;
    }
    chart.options.scales.x = getXAxisOptions();
    chart.options.scales.y = getYAxisOptions();
    chart.update();
  });

  const periodToInterval = {
    hour:  "1m",
    day:   "1h",
    week:  "none",
    month: "none",
    year:  "none"
  };

  const periodToPeriodCode = {
    hour:  "1h",
    day:   "1d",
    week:  "none",
    month: "none",
    year:  "none"
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
    
    setTimeout(setSwiperHeight, 100);
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
    
    setTimeout(setSwiperHeight, 100);
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
      updateUI();
    });
  });
  
  tippy('.graph__button--disabled', {
    content: "Currently not available",
    arrow: true,
    animation: 'fade',
  });

  document.querySelectorAll('[data-interval]').forEach(button => {
    button.addEventListener('click', event => {
      if (button.classList.contains('graph__button--disabled')) {
        event.preventDefault();
        event.stopPropagation();
        return;
      }

      selectedDate = button.dataset.interval;
      activeInterval();
    });
  });

  activeInterval();
  activeSelect();

  addRippleEffectToMetalButtons();
  
  updateNewsContent();
  
  setSwiperHeight();

  let newsButton = document.querySelector('.swiper-button-open');
  let newsSection = document.querySelector('.news');
  let newsButtonClose = document.querySelector('.swiper-button-close');

  newsButton.addEventListener('click', function(e) {
    e.stopPropagation();
    newsSection.style.left = '12px';
  });

  document.addEventListener('click', function(e) {
    if (!newsSection.contains(e.target)) {
      newsSection.style.left = '';
    }
  });

  newsButtonClose.addEventListener('click', function(e) {
    e.stopPropagation();
    newsSection.style.left = '';
  });
  
  let burgerHeaderButton = document.querySelector('.header__burger');
  let burgerHeaderClose = document.querySelector('.header__burger--close');
  let burgerHeaderMenu = document.querySelector('.header__burger-block');

  burgerHeaderButton.addEventListener('click', function(e) {
    e.stopPropagation();
    burgerHeaderMenu.style.left = '20%';
  });

  document.addEventListener('click', function(e) {
    if (!burgerHeaderMenu.contains(e.target)) {
      burgerHeaderMenu.style.left = '';
    }
  });

  burgerHeaderClose.addEventListener('click', function(e) {
    e.stopPropagation();
    burgerHeaderMenu.style.left = '';
  });

  window.addEventListener('resize', setSwiperHeight);

  setupCustomSelects();
  syncCustomSelects();
  patchButtonSync();
});

function setSwiperHeight() {
  const graphSection = document.querySelector('.graph');
  const swiperElement = document.querySelector('.swiper');
  
  if (graphSection && swiperElement) {
    const graphHeight = graphSection.offsetHeight;
    swiperElement.style.height = graphHeight + 'px';
    
    if (swiper) {
      swiper.update();
    }
  }
}

function updateNewsContent() {
  const newsData = metalNews[selectedMetal];
  const swiperWrapper = document.querySelector('.swiper-wrapper');
  
  swiperWrapper.innerHTML = '';
  
  newsData.forEach(newsItem => {
    const slide = document.createElement('div');
    slide.className = 'swiper-slide';
    slide.innerHTML = `
      <div>
        <p class="news__info">
          ${newsItem.info}
        </p>
        <p class="news__date">
          ${newsItem.date}
        </p>
      </div>
    `;
    swiperWrapper.appendChild(slide);
  });
  
  if (swiper) {
    swiper.update();
  }
}

function activeInterval() {
  document.querySelectorAll('[data-interval]').forEach(button => {
    const isSelected = button.dataset.interval === selectedDate;

    if (isSelected) {
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

function updateUI() {
  let title = document.querySelector('.graph__title');
  title.textContent = metalNames[selectedMetal];

  let newsTitle = document.querySelector('.news__title--metal');
  newsTitle.textContent = metalNames[selectedMetal];

  updateNewsContent();
  
  setTimeout(setSwiperHeight, 100);
}

function setupCustomSelects() {
  function closeAllSelects(except) {
    document.querySelectorAll('.custom-select').forEach(sel => {
      if (sel !== except) sel.classList.remove('open');
    });
  }

  document.querySelectorAll('.custom-select').forEach(select => {
    const selected = select.querySelector('.custom-select__selected');
    const options = select.querySelector('.custom-select__options');
    const optionItems = select.querySelectorAll('.custom-select__option');

    select.addEventListener('click', function(e) {
      e.stopPropagation();
      if (select.classList.contains('open')) {
        select.classList.remove('open');
      } else {
        closeAllSelects(select);
        select.classList.add('open');
      }
    });

    optionItems.forEach(option => {
      option.addEventListener('click', function(e) {
        e.stopPropagation();
        if (option.classList.contains('custom-select__option--disabled')) return;
        selected.textContent = option.textContent;
        optionItems.forEach(opt => opt.classList.remove('selected'));
        option.classList.add('selected');
        select.classList.remove('open');

        if (select.id === 'period-select') {
          const val = option.getAttribute('data-value');
          const btn = document.querySelector('.graph__button[data-interval="' + val + '"]');
          if (btn && !btn.classList.contains('graph__button--disabled')) btn.click();
        } 
        
        else if (select.id === 'type-select') {
          const val = option.getAttribute('data-value');
          const btn = document.querySelector('.graph__button[data-select="' + val + '"]');
          if (btn) btn.click();
        }
      });
    });
  });

  document.addEventListener('click', function() {
    closeAllSelects();
  });
}

function syncCustomSelects() {
  const periodBtn = document.querySelector('.graph__button.graph__button--active[data-interval]');
  if (periodBtn) {
    const val = periodBtn.getAttribute('data-interval');
    const select = document.getElementById('period-select');
    if (select) {
      const selected = select.querySelector('.custom-select__selected');
      const option = select.querySelector('.custom-select__option[data-value="' + val + '"]');
      if (option) {
        selected.textContent = option.textContent;
        select.querySelectorAll('.custom-select__option').forEach(opt => opt.classList.remove('selected'));
        option.classList.add('selected');
      }
    }
  }

  const typeBtn = document.querySelector('.graph__button.active[data-select], .graph__button.graph__button--active[data-select]');
  if (typeBtn) {
    const val = typeBtn.getAttribute('data-select');
    const select = document.getElementById('type-select');
    if (select) {
      const selected = select.querySelector('.custom-select__selected');
      const option = select.querySelector('.custom-select__option[data-value="' + val + '"]');
      if (option) {
        selected.textContent = option.textContent;
        select.querySelectorAll('.custom-select__option').forEach(opt => opt.classList.remove('selected'));
        option.classList.add('selected');
      }
    }
  }
}

function patchButtonSync() {
  document.querySelectorAll('.graph__button').forEach(button => {
    button.addEventListener('click', () => {
      setTimeout(syncCustomSelects, 0);
    });
  });
}