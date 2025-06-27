<h1 align="center">Metalytics</h1>

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20progress-yellow.svg">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img src="https://img.shields.io/github/issues/IU-Capstone-Project-2025/Metalytics" alt="GitHub issues">
  <img src="https://img.shields.io/github/last-commit/IU-Capstone-Project-2025/Metalytics" alt="Last commit">
</p>

## 📌 Overview

The project idea is to develop an analytical system for forecasting the prices of specific metals traded on the Russian financial market.  
The project will incorporate study of modern approaches of precious metal price forecasting and trends to propose a set of instruments suitable for market analysis of specific metal types.  
Additionally, the system will generate reports explaining price fluctuations and potential market volatility based on news, sanctions, macroeconomic indicators, and other external factors.

---


## 📖 Contents
- [📌 Overview](#-overview)
- [🚀 Tech Stack](#-tech-stack)
- [📁 Folder Structure](#-folder-structure)
- [🔥 Getting Started](#-getting-started)
- [📍 Waypoints](#-waypoints)
- [👥 Team](#-team)
- [📈 Roadmap](#-roadmap)
- [📝 License](#-license)

---

## 🚀 Tech Stack

| Category       | Tools / Libraries                              | Why we chose them                             |
|----------------|------------------------------------------------|-----------------------------------------------|
| ML / Baseline  | scikit-learn                                 | Lightweight, well-documented, great for fast prototyping |
| Deep Learning  | Keras, TensorFlow, PyTorch, Theano     | For advanced modeling |
| Metaheuristics | DEAP, scikit-opt, pyswarmpackage         | For experimenting with alternative optimizers |
| Backend        | FastAPI                                      | Fast, modern, built-in OpenAPI docs           |
| Frontend       | HTML, CSS, JavaScript                    | Simple static frontend |
| Data Processing| pandas, NumPy                              | Standard tools for loading and transforming data |
| Infrastructure | Docker, docker-compose                     | Reproducibility, unified local setup          |

## 📁 Folder Structure

```
Metalytics/
├── frontend/           # Client-side application
├── backend/            # Server-side logic and API handling
├── ml/                 # Machine learning models, training, and inference scripts
├── docker-compose.yml  # Orchestration file for running all services together
├── .gitignore          # Git exclusion rules
└── README.md           # Project overview and instructions
```

---

## 🔥 Getting Started

Follow the steps below to run the project locally using Docker or manually.

### ⚙️ Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed and running
- Python 3.10+
- (Optional) Live server if running without Docker

### 📥 Step 1 — Clone the Repository

In your terminal:

```bash
git clone https://github.com/IU-Capstone-Project-2025/Metalytics.git
cd Metalytics
```

### 💻 Step 2 — Run with Docker (Recommended)

```bash
docker-compose up --build
```
This will:
- Start the **FastAPI backend** at [http://localhost:8000](http://localhost:8000)
- Start the **Frontend** at [http://localhost:3000](http://localhost:3000)

#### 📋 Managing the Container

To view logs from the running container:
```bash
docker compose logs -f
```

To stop the container without removing it:
```bash
docker compose stop
```

To stop and remove the container:
```bash
docker compose down
```

To rebuild the image after making changes to the code:
```bash
docker compose up -d --build
```

### 🛠 Alternative — Run Manually (Without Docker)
- Clone the repository (follow step 1).
- Make sure you are in the project root folder.
- Follow the steps bellow.

#### ▶️ Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
#### 🌐 Frontend
Open frontend/index.html directly in your browser or use a local server (ex. "Live Server" in VS Code)

---

## 📍 Waypoints

| Method | Endpoint                    | Function                                                              |
| -----  | --------------------------- | --------------------------------------------------------------------- |
| GET    | /metals                     | Get a list of available metals                                        |
| GET    | /historical_data/{metal_id} | Get a historical data.                                                |
| GET    | /forecast/{metal_id}        | Get a metal price forecast                                            |
| GET    | /forecast/{metal_id}/days   | Forecast for N days ahead                                             |
| GET    | /health                     | Checking if the backend is working                                    |
| GET    | /version                    | API/model version                                                     |          

## 👥 Team


|       **Name**       |         **Role**          |              **Responsibilities**               |      **Email**      |
|:--------------------:|:-------------------------:|:-----------------------------------------------:|:--------------------------:|
| Ilya Grigorev        | ML Engineer               | R&D, Filtration design, Model Selection, and ML team coordination |         il.grigorev@innopolis.university                 |
| Farit Sharafutdinov  | ML Engineer               | Data collection, preprocessing, and delivery     |           f.sharafutdinov@innopolis.university               |
| Rail Sharipov        | ML Engineer               | R&D, Explaratory data analysis, and feature engineering                                 |            ra.sharipov@innopolis.university              |
| Askar Kadyrgulov     | Backend Developer         | Development and Operations for backend and ML, scraping functionality                   |            a.kadyrgulov@innopolis.university              |
| Nikita Solomennikov  | Designer                  | Creating design for frontend                     |              n.solomennikov@innopolis.unisersity            |
| Vladimir Toporkov    | Frontend Developer / Team Lead | Frontend development, team coordination       |               v.toporkov@innopolis.university           |

---

## 📈 Roadmap

- [x] Project structure setup
- [ ] Frontend and backend boilerplates
- [ ] Data collection and preprocessing
- [ ] ML baseline model
- [ ] Explainability reports (news, sanctions, macro trends)
- [ ] UI integration

## 📝 License

This project is licensed under the [MIT License](LICENSE).
