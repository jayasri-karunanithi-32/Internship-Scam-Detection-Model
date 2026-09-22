/**
 * Internship Scam Detection - Frontend JavaScript
 * Handles tab switching, input handling, API calls, and result display.
 */

const API_BASE_URL = "http://localhost:8000";

let currentTab = "text";
let selectedFile = null;

/* ============================================
   Tab Management
   ============================================ */

function switchTab(tab) {
    currentTab = tab;

    // Update tab buttons
    document.querySelectorAll(".tab-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.tab === tab);
    });

    // Update tab panes
    document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.toggle("active", pane.id === `tab-${tab}`);
    });
}

/* ============================================
   File Upload Handling
   ============================================ */

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file size (10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert("File size exceeds 10MB. Please choose a smaller file.");
        return;
    }

    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = function (e) {
        document.getElementById("preview-image").src = e.target.result;
        document.getElementById("file-name").textContent = file.name;
        document.getElementById("file-preview").style.display = "flex";
        document.getElementById("upload-area").style.display = "none";
    };
    reader.readAsDataURL(file);
}

function removeFile() {
    selectedFile = null;
    document.getElementById("image-input").value = "";
    document.getElementById("file-preview").style.display = "none";
    document.getElementById("upload-area").style.display = "block";
}

// Drag and drop support
document.addEventListener("DOMContentLoaded", function () {
    const uploadArea = document.getElementById("upload-area");

    if (uploadArea) {
        uploadArea.addEventListener("dragover", function (e) {
            e.preventDefault();
            uploadArea.classList.add("drag-over");
        });

        uploadArea.addEventListener("dragleave", function (e) {
            e.preventDefault();
            uploadArea.classList.remove("drag-over");
        });

        uploadArea.addEventListener("drop", function (e) {
            e.preventDefault();
            uploadArea.classList.remove("drag-over");
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                document.getElementById("image-input").files = files;
                handleFileSelect({ target: { files: files } });
            }
        });
    }
});

/* ============================================
   Analysis Functions
   ============================================ */

async function analyzeInput() {
    const analyzeBtn = document.getElementById("analyze-btn");

    // Validate input
    let hasInput = false;
    if (currentTab === "text") {
        hasInput = document.getElementById("text-input").value.trim().length > 0;
    } else if (currentTab === "url") {
        hasInput = document.getElementById("url-input").value.trim().length > 0;
    } else if (currentTab === "image") {
        hasInput = selectedFile !== null;
    }

    if (!hasInput) {
        alert("Please provide input to analyze.");
        return;
    }

    // Show loading, hide results/errors
    showSection("loading");
    analyzeBtn.disabled = true;

    try {
        let response;

        if (currentTab === "text") {
            response = await analyzeText(
                document.getElementById("text-input").value.trim()
            );
        } else if (currentTab === "url") {
            response = await analyzeURL(
                document.getElementById("url-input").value.trim()
            );
        } else if (currentTab === "image") {
            response = await analyzeImage(selectedFile);
        }

        if (response.success) {
            displayResults(response.result);
        } else {
            showError(response.error || "Analysis failed. Please try again.");
        }
    } catch (error) {
        console.error("Analysis error:", error);
        showError(
            error.message || "Failed to connect to the server. Please ensure the backend is running."
        );
    } finally {
        analyzeBtn.disabled = false;
    }
}

async function analyzeText(text) {
    const formData = new FormData();
    formData.append("text", text);

    const response = await fetch(`${API_BASE_URL}/api/analyze/text`, {
        method: "POST",
        body: formData,
    });

    return await response.json();
}

async function analyzeURL(url) {
    const formData = new FormData();
    formData.append("url", url);

    const response = await fetch(`${API_BASE_URL}/api/analyze/url`, {
        method: "POST",
        body: formData,
    });

    return await response.json();
}

async function analyzeImage(file) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/api/analyze/image`, {
        method: "POST",
        body: formData,
    });

    return await response.json();
}

/* ============================================
   Display Functions
   ============================================ */

function displayResults(result) {
    const resultCard = document.getElementById("result-card");
    const isGenuine = result.prediction === "Genuine";
    const cssClass = isGenuine ? "genuine" : "fraudulent";

    // Set card class
    resultCard.className = `result-card ${cssClass}`;

    // Set icon
    document.getElementById("result-icon").innerHTML = isGenuine
        ? '<svg viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5" width="36" height="36"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2.5" width="36" height="36"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';

    // Set label
    document.getElementById("result-label").textContent = isGenuine
        ? "Genuine Internship"
        : "Fraudulent Internship";

    // Set model info
    document.getElementById("result-model-info").textContent =
        `Classified by ${result.model_used} (Model accuracy: ${result.model_accuracy}%)`;

    // Animate confidence ring
    const confidence = result.confidence;
    const circumference = 2 * Math.PI * 52; // 326.73
    const offset = circumference - (confidence / 100) * circumference;

    const ringFill = document.getElementById("ring-fill");
    ringFill.style.strokeDasharray = circumference;
    ringFill.style.strokeDashoffset = circumference;

    // Trigger animation
    setTimeout(() => {
        ringFill.style.strokeDashoffset = offset;
    }, 100);

    // Animate confidence value
    animateValue("confidence-value", 0, confidence, 1500);

    // Set explanation
    const explanationContent = document.getElementById("explanation-content");
    explanationContent.innerHTML = formatExplanation(result.explanation);

    // Set processing steps
    const stepsGrid = document.getElementById("steps-grid");
    if (result.steps) {
        stepsGrid.innerHTML = `
            <div class="step-item">
                <div class="step-label">Original Text Length</div>
                <div class="step-value">${result.steps.original.length} characters</div>
            </div>
            <div class="step-item">
                <div class="step-label">After Cleaning</div>
                <div class="step-value">${result.steps.cleaned.length} characters</div>
            </div>
            <div class="step-item">
                <div class="step-label">Tokens Generated</div>
                <div class="step-value">${result.steps.tokens_count} tokens</div>
            </div>
            <div class="step-item">
                <div class="step-label">After Stopword Removal</div>
                <div class="step-value">${result.steps.after_stopword_removal} tokens</div>
            </div>
        `;
    }

    showSection("results");
}

function formatExplanation(text) {
    if (!text) return "";

    // Convert markdown-like bold to HTML
    let html = text
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");

    return html;
}

function animateValue(elementId, start, end, duration) {
    const element = document.getElementById(elementId);
    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + range * eased;

        element.textContent = `${current.toFixed(1)}%`;

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

function showError(message) {
    document.getElementById("error-message").textContent = message;
    showSection("error");
}

function showSection(section) {
    document.getElementById("loading-section").style.display =
        section === "loading" ? "block" : "none";
    document.getElementById("results-section").style.display =
        section === "results" ? "block" : "none";
    document.getElementById("error-section").style.display =
        section === "error" ? "block" : "none";
}

function resetAnalysis() {
    showSection(null);
    // Scroll to top
    window.scrollTo({ top: 0, behavior: "smooth" });
}
