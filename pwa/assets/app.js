const photo = document.querySelector("#photo");
const preview = document.querySelector("#preview");
const detect = document.querySelector("#detect");
const status = document.querySelector("#status");

photo.addEventListener("change", () => {
  const file = photo.files[0];
  if (!file) {
    preview.removeAttribute("src");
    preview.style.display = "none";
    detect.disabled = true;
    return;
  }

  preview.src = URL.createObjectURL(file);
  preview.style.display = "block";
  detect.disabled = false;
});

detect.addEventListener("click", async () => {
  const file = photo.files[0];
  if (!file) return;

  detect.disabled = true;
  status.textContent = "Analyzing photo...";
  const form = new FormData();
  form.append("file", file);

  try {
    const response = await fetch("/api/predict", { method: "POST", body: form });
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Prediction failed");
    }

    const result = await response.json();
    document.querySelector("#species").value = result.species;
    document.querySelector("#breed").value = result.breed;
    document.querySelector("#color").value = result.color;
    document.querySelector("#size").value = result.size;
    status.textContent =
      `Breed confidence: ${Math.round(result.breed_confidence * 100)}% · ` +
      `Color estimate: ${Math.round(result.color_confidence * 100)}%`;
  } catch (error) {
    status.textContent = error.message;
  } finally {
    detect.disabled = false;
  }
});

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/assets/service-worker.js").catch(() => {});
  });
}
