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