/**
 * AgriScan AI - Frontend Controller
 * Crop Disease Detection Using Convolutional Neural Networks
 * Author: Virat Dagar | ABES Engineering College
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const previewCard = document.getElementById("previewCard");
    const imagePreview = document.getElementById("imagePreview");
    const previewFileName = document.getElementById("previewFileName");
    const previewFileSize = document.getElementById("previewFileSize");
    const removeBtn = document.getElementById("removeBtn");
    const analyzeBtn = document.getElementById("analyzeBtn");

    const uploadView = document.getElementById("uploadView");
    const loadingView = document.getElementById("loadingView");
    const loadingImagePreview = document.getElementById("loadingImagePreview");
    const resultView = document.getElementById("resultView");

    const resCondition = document.getElementById("resCondition");
    const resCrop = document.getElementById("resCrop");
    const resStatusBadge = document.getElementById("resStatusBadge");
    const resConfidence = document.getElementById("resConfidence");
    const resConfidenceBar = document.getElementById("resConfidenceBar");
    const resImage = document.getElementById("resImage");
    const topPredictionsList = document.getElementById("topPredictionsList");
    const resCause = document.getElementById("resCause");
    const resSymptoms = document.getElementById("resSymptoms");
    const resManagement = document.getElementById("resManagement");
    const tryAgainBtn = document.getElementById("tryAgainBtn");

    const alertBox = document.getElementById("alertBox");
    const alertMessage = document.getElementById("alertMessage");

    let currentFile = null;

    // Allowed Extensions & Max Size
    const ALLOWED_EXTS = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    const MAX_SIZE_MB = 16;

    // --- Drag and Drop Handlers ---
    ["dragenter", "dragover"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt.files && dt.files.length > 0) {
            handleSelectedFile(dt.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleSelectedFile(e.target.files[0]);
        }
    });

    // --- File Validation & Preview ---
    function handleSelectedFile(file) {
        hideAlert();

        if (!ALLOWED_EXTS.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
            showAlert("Please upload a valid image file (JPG, PNG, or WEBP).");
            return;
        }

        if (file.size > MAX_SIZE_MB * 1024 * 1024) {
            showAlert(`File exceeds the maximum size limit of ${MAX_SIZE_MB} MB.`);
            return;
        }

        currentFile = file;
        previewFileName.textContent = file.name;
        previewFileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + " MB";

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            loadingImagePreview.src = e.target.result;
            previewCard.classList.remove("hidden");
            dropZone.classList.add("hidden");
        };
        reader.readAsDataURL(file);
    }

    // --- Remove File Action ---
    removeBtn.addEventListener("click", () => {
        resetFileInput();
    });

    function resetFileInput() {
        currentFile = null;
        fileInput.value = "";
        imagePreview.src = "";
        previewCard.classList.add("hidden");
        dropZone.classList.remove("hidden");
    }

    // --- Analyze Leaf Action ---
    analyzeBtn.addEventListener("click", async () => {
        if (!currentFile) {
            showAlert("Please select a leaf image first.");
            return;
        }

        // Switch to loading view
        uploadView.classList.add("hidden");
        loadingView.classList.remove("hidden");
        resultView.classList.add("hidden");

        const formData = new FormData();
        formData.append("image", currentFile);

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "An error occurred while analyzing the image.");
            }

            // Display Prediction Results
            displayResults(data);

        } catch (err) {
            let msg = err.message || "Network error. Please try again.";
            if (msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
                msg = "Failed to connect to backend server. Please ensure Flask app is running (python app.py).";
            }
            showAlert(msg);
            loadingView.classList.add("hidden");
            uploadView.classList.remove("hidden");
        }
    });

    // --- Display Result Helper ---
    function displayResults(data) {
        loadingView.classList.add("hidden");
        resultView.classList.remove("hidden");

        // Crop & Condition
        resCondition.textContent = data.condition;
        resCrop.innerHTML = `<i class="fa-solid fa-seedling"></i> ${data.crop}`;

        // Status Badge (Healthy vs Pathogen Detected)
        if (data.is_healthy) {
            resStatusBadge.className = "status-badge healthy";
            resStatusBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Healthy Leaf`;
        } else {
            resStatusBadge.className = "status-badge diseased";
            resStatusBadge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Pathogen Detected`;
        }

        // Confidence
        resConfidence.textContent = `${data.confidence}%`;
        resConfidenceBar.style.width = `${Math.min(100, Math.max(10, data.confidence))}%`;

        // Image with fallback
        resImage.onerror = () => {
            resImage.src = imagePreview.src || "/static/samples/tomato_bacterial_spot.jpg";
        };
        resImage.src = data.image_url || imagePreview.src;

        // Top 3 Probabilities
        topPredictionsList.innerHTML = "";
        if (data.top_predictions && data.top_predictions.length > 0) {
            data.top_predictions.forEach(p => {
                const item = document.createElement("div");
                item.className = "pred-item";
                item.innerHTML = `
                    <div class="pred-header">
                        <span>${p.crop} &bull; ${p.condition}</span>
                        <span>${p.probability}%</span>
                    </div>
                    <div class="pred-track">
                        <div class="pred-bar" style="width: ${p.probability}%"></div>
                    </div>
                `;
                topPredictionsList.appendChild(item);
            });
        }

        // Advisory Data
        if (data.advisory) {
            resCause.textContent = data.advisory.cause || "Pathological Leaf Condition";
            resSymptoms.textContent = data.advisory.symptoms || "Foliage discoloration or lesions observed.";
            resManagement.textContent = data.advisory.management || "Consult agricultural extension guidelines.";
        }

        // Scroll smoothly to results
        resultView.scrollIntoView({ behavior: "smooth" });
    }

    // --- Try Again / Reset ---
    tryAgainBtn.addEventListener("click", () => {
        resultView.classList.add("hidden");
        uploadView.classList.remove("hidden");
        resetFileInput();
        window.scrollTo({ top: 0, behavior: "smooth" });
    });

    // --- Alert Utilities ---
    function showAlert(msg) {
        alertMessage.textContent = msg;
        alertBox.classList.remove("hidden");
    }

    window.hideAlert = function() {
        alertBox.classList.add("hidden");
    };
});
