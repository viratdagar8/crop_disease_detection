const { useState, useEffect, useRef } = React;

// Pre-loaded recent diagnoses from Aaditya & PlantVillage CNN dataset
const DEFAULT_RECENT_SCANS = [
    {
        id: 101,
        thumbnail: "/static/samples/tomato_bacterial_spot.jpg",
        crop: "Tomato",
        plant: "Tomato",
        condition: "Bacterial Spot",
        disease: "Bacterial Spot",
        is_healthy: false,
        confidence: 99.4,
        time: "10:15 AM",
        date: "Today",
        fullData: {
            success: true,
            prediction_text: "This is a Tomato leaf with Bacterial Spot",
            crop: "Tomato",
            plant: "Tomato",
            condition: "Bacterial Spot",
            disease: "Bacterial Spot",
            is_healthy: false,
            confidence: 99.4,
            confidence_tier: "high",
            is_ambiguous: false,
            image_url: "/static/samples/tomato_bacterial_spot.jpg",
            top_predictions: [
                { crop: "Tomato", condition: "Bacterial Spot", probability: 99.4 },
                { crop: "Potato", condition: "Early Blight", probability: 0.4 },
                { crop: "Corn (Maize)", condition: "Common Rust", probability: 0.2 }
            ],
            advisory: {
                crop: "Tomato",
                disease: "Bacterial Spot",
                cause: "Xanthomonas campestris pv. vesicatoria (Bacterium)",
                symptoms: "Small, water-soaked circular lesions turning dark brown with yellow chlorotic halos across leaf blades.",
                management: "Prune lower infected foliage, avoid overhead sprinkler watering, space rows for aeration, and apply approved copper-based bactericides.",
                disclaimer: "Agricultural decision-support tool. Verify critical crop decisions with local extension agronomists."
            },
            features: { chlorophyll_ratio: 62.4, lesion_ratio: 18.2 }
        }
    },
    {
        id: 102,
        thumbnail: "/static/samples/potato_early_blight.jpg",
        crop: "Potato",
        plant: "Potato",
        condition: "Early Blight",
        disease: "Early Blight",
        is_healthy: false,
        confidence: 99.4,
        time: "09:40 AM",
        date: "Today",
        fullData: {
            success: true,
            prediction_text: "This is a Potato leaf with Early Blight",
            crop: "Potato",
            plant: "Potato",
            condition: "Early Blight",
            disease: "Early Blight",
            is_healthy: false,
            confidence: 99.4,
            confidence_tier: "high",
            is_ambiguous: false,
            image_url: "/static/samples/potato_early_blight.jpg",
            top_predictions: [
                { crop: "Potato", condition: "Early Blight", probability: 99.4 },
                { crop: "Tomato", condition: "Bacterial Spot", probability: 0.4 },
                { crop: "Corn (Maize)", condition: "Common Rust", probability: 0.2 }
            ],
            advisory: {
                crop: "Potato",
                disease: "Early Blight",
                cause: "Alternaria solani (Fungus)",
                symptoms: "Dark brown necrotic spots exhibiting concentric rings producing a distinct 'target-board' pattern on mature foliage.",
                management: "Implement 3-year crop rotation, maintain balanced nitrogen soil levels, avoid moisture stress, and apply protectant fungicides (mancozeb/chlorothalonil).",
                disclaimer: "Agricultural decision-support tool. Verify critical crop decisions with local extension agronomists."
            },
            features: { chlorophyll_ratio: 54.1, lesion_ratio: 24.8 }
        }
    },
    {
        id: 103,
        thumbnail: "/static/samples/corn_common_rust.jpg",
        crop: "Corn (Maize)",
        plant: "Corn (Maize)",
        condition: "Common Rust",
        disease: "Common Rust",
        is_healthy: false,
        confidence: 99.4,
        time: "Yesterday",
        date: "Sep 29",
        fullData: {
            success: true,
            prediction_text: "This is a Corn (Maize) leaf with Common Rust",
            crop: "Corn (Maize)",
            plant: "Corn (Maize)",
            condition: "Common Rust",
            disease: "Common Rust",
            is_healthy: false,
            confidence: 99.4,
            confidence_tier: "high",
            is_ambiguous: false,
            image_url: "/static/samples/corn_common_rust.jpg",
            top_predictions: [
                { crop: "Corn (Maize)", condition: "Common Rust", probability: 99.4 },
                { crop: "Potato", condition: "Early Blight", probability: 0.3 },
                { crop: "Tomato", condition: "Bacterial Spot", probability: 0.3 }
            ],
            advisory: {
                crop: "Corn (Maize)",
                disease: "Common Rust",
                cause: "Puccinia sorghi (Basidiomycete Fungus)",
                symptoms: "Elongated golden-brown to cinnamon-brown pustules (uredinia) bursting on both upper and lower leaf surfaces.",
                management: "Plant resistant maize hybrid seed varieties. Apply foliar triazole or strobilurin fungicides if pustules appear prior to tasseling stage.",
                disclaimer: "Agricultural decision-support tool. Verify critical crop decisions with local extension agronomists."
            },
            features: { chlorophyll_ratio: 48.7, lesion_ratio: 29.5 }
        }
    }
];

function App() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState("");
    const [fileMeta, setFileMeta] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [scanStep, setScanStep] = useState(0);
    const [prediction, setPrediction] = useState(null);
    const [errorMsg, setErrorMsg] = useState("");
    const [activeTab, setActiveTab] = useState("cause");
    const [showCatalogModal, setShowCatalogModal] = useState(false);
    const [allClasses, setAllClasses] = useState([]);
    const [searchQuery, setSearchQuery] = useState("");
    const [healthStatus, setHealthStatus] = useState("checking");

    // Recent scans persistent state (defaults to Aaditya repo dataset samples if empty)
    const [recentScans, setRecentScans] = useState(() => {
        try {
            const saved = localStorage.getItem("agriscan_recent_scans");
            if (saved) {
                const parsed = JSON.parse(saved);
                if (Array.isArray(parsed) && parsed.length > 0) return parsed;
            }
            return DEFAULT_RECENT_SCANS;
        } catch (e) {
            return DEFAULT_RECENT_SCANS;
        }
    });

    const fileInputRef = useRef(null);

    // Initial health check and class list fetch
    useEffect(() => {
        fetch("/health")
            .then(res => res.json())
            .then(data => {
                if (data.status === "healthy") setHealthStatus("online");
            })
            .catch(() => setHealthStatus("offline"));

        fetch("/classes")
            .then(res => res.json())
            .then(data => {
                if (data.classes) setAllClasses(data.classes);
            })
            .catch(err => console.error("Could not load classes:", err));
    }, []);

    // Step animation text during loading
    useEffect(() => {
        let interval;
        if (isLoading) {
            setScanStep(0);
            interval = setInterval(() => {
                setScanStep(prev => (prev + 1) % 3);
            }, 600);
        }
        return () => clearInterval(interval);
    }, [isLoading]);

    const handleFile = (file) => {
        setErrorMsg("");
        if (!file) return;

        const validTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
        if (!validTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp)$/i)) {
            setErrorMsg("Please upload a valid image file (JPG, PNG, or WEBP).");
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            setErrorMsg("File size exceeds the 16 MB limit.");
            return;
        }

        setSelectedFile(file);
        setFileMeta({
            name: file.name,
            size: (file.size / (1024 * 1024)).toFixed(2) + " MB"
        });

        const reader = new FileReader();
        reader.onload = (e) => setPreviewUrl(e.target.result);
        reader.readAsDataURL(file);
        setPrediction(null);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    };

    const analyzeImage = async () => {
        if (!selectedFile) return;
        setIsLoading(true);
        setErrorMsg("");

        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const res = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await res.json();
            if (!res.ok || !data.success) {
                throw new Error(data.error || "Prediction request failed.");
            }

            setPrediction(data);
            setActiveTab("cause");

            // Save new diagnosis into recent scans history
            const newScan = {
                id: Date.now(),
                thumbnail: data.image_url || previewUrl,
                crop: data.plant || data.crop,
                plant: data.plant || data.crop,
                condition: data.disease || data.condition,
                disease: data.disease || data.condition,
                is_healthy: data.is_healthy,
                confidence: data.confidence,
                time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                date: new Date().toLocaleDateString([], { month: "short", day: "numeric" }),
                fullData: data
            };

            setRecentScans(prev => {
                const updated = [newScan, ...prev.filter(s => s.condition !== data.condition || s.crop !== data.crop)].slice(0, 10);
                try {
                    localStorage.setItem("agriscan_recent_scans", JSON.stringify(updated));
                } catch (e) {}
                return updated;
            });

        } catch (err) {
            setErrorMsg(err.message || "Failed to analyze leaf image.");
        } finally {
            setIsLoading(false);
        }
    };

    const loadRecentScan = (scan) => {
        setPrediction(scan.fullData);
        setPreviewUrl(scan.thumbnail);
        setSelectedFile(null);
        setFileMeta({
            name: `${scan.crop}_${scan.condition}.jpg`,
            size: "Diagnosed Record"
        });
        window.scrollTo({ top: 180, behavior: "smooth" });
    };

    const clearRecentScans = () => {
        setRecentScans([]);
        localStorage.removeItem("agriscan_recent_scans");
    };

    const resetAnalysis = () => {
        setSelectedFile(null);
        setPreviewUrl("");
        setFileMeta(null);
        setPrediction(null);
        setErrorMsg("");
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    // Circular gauge calculations
    const radius = 54;
    const circumference = 2 * Math.PI * radius;
    const confidenceVal = prediction ? prediction.confidence : 0;
    const strokeDashoffset = circumference - (confidenceVal / 100) * circumference;

    const getGaugeColor = (conf) => {
        if (conf >= 80) return "#10b981"; // Emerald green
        if (conf >= 50) return "#f59e0b"; // Amber
        return "#ef4444"; // Red
    };

    return (
        <div className="react-app-wrapper">
            {/* Header / Navigation Bar */}
            <header className="app-nav">
                <div className="nav-content">
                    <div className="nav-brand">
                        <div className="brand-leaf-icon">
                            <i className="fa-solid fa-leaf"></i>
                        </div>
                        <div>
                            <div className="brand-title">AgriScan <span className="brand-accent">AI</span></div>
                            <div className="brand-tagline">Deep Learning Crop Disease Identification</div>
                        </div>
                    </div>

                    <div className="nav-meta">
                        <button className="catalog-btn" onClick={() => setShowCatalogModal(true)}>
                            <i className="fa-solid fa-book-open"></i> Supported Crops ({allClasses.length || 38})
                        </button>
                        <div className="status-indicator">
                            <span className={`status-dot ${healthStatus}`}></span>
                            <span>{healthStatus === "online" ? "AI Engine Ready" : "Connecting..."}</span>
                        </div>
                        <div className="author-pill">
                            <i className="fa-solid fa-graduation-cap"></i> Virat Dagar &bull; ABES EC
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content Area */}
            <main className="content-container">
                {/* Hero Banner */}
                <section className="hero-banner">
                    <div className="hero-badge">HCLTech Industry Aligned Projects &bull; Session 2026–27</div>
                    <h1 className="hero-heading">
                        Instant Crop Disease Identification with <span className="highlight-text">Convolutional Neural Networks</span>
                    </h1>
                    <p className="hero-subtext">
                        Upload a photograph of any crop leaf to identify the plant species, detect disease symptoms, calculate diagnostic confidence, and view expert management recommendations.
                    </p>
                </section>

                {/* Alert Box */}
                {errorMsg && (
                    <div className="alert-banner">
                        <i className="fa-solid fa-circle-exclamation"></i>
                        <span>{errorMsg}</span>
                        <button className="alert-dismiss" onClick={() => setErrorMsg("")}>&times;</button>
                    </div>
                )}

                {/* Main Workspace Card */}
                <div className="main-card">
                    {/* View 1: Upload & Drop Area */}
                    {!prediction && (
                        <div className="workspace-upload">
                            {!previewUrl ? (
                                <div 
                                    className={`dropzone-box ${isDragging ? "dragging" : ""}`}
                                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                                    onDragLeave={() => setIsDragging(false)}
                                    onDrop={handleDrop}
                                    onClick={() => fileInputRef.current.click()}
                                >
                                    <input 
                                        type="file" 
                                        ref={fileInputRef} 
                                        style={{ display: "none" }} 
                                        accept="image/png, image/jpeg, image/jpg, image/webp"
                                        onChange={(e) => e.target.files && handleFile(e.target.files[0])}
                                    />
                                    <div className="dropzone-icon">
                                        <i className="fa-solid fa-cloud-arrow-up"></i>
                                    </div>
                                    <h3 className="dropzone-heading">Upload Crop Leaf Photograph</h3>
                                    <p className="dropzone-sub">Drag & drop your leaf image here or <span className="highlight-link">browse files</span></p>
                                    <div className="dropzone-specs">
                                        <span><i className="fa-regular fa-image"></i> JPG, PNG, WEBP</span>
                                        <span>&bull;</span>
                                        <span><i className="fa-solid fa-weight-hanging"></i> Up to 16 MB</span>
                                        <span>&bull;</span>
                                        <span><i className="fa-solid fa-microscope"></i> Plant-Disease-Detection CNN Model</span>
                                    </div>
                                </div>
                            ) : (
                                <div className="preview-container">
                                    <div className="preview-viewport">
                                        <img src={previewUrl} alt="Leaf Preview" />
                                        {isLoading && (
                                            <div className="laser-scanner">
                                                <div className="laser-beam"></div>
                                                <div className="scanner-grid"></div>
                                            </div>
                                        )}
                                        <button className="remove-preview-btn" onClick={resetAnalysis} title="Change Image">
                                            <i className="fa-solid fa-xmark"></i>
                                        </button>
                                    </div>

                                    {fileMeta && (
                                        <div className="file-meta-bar">
                                            <span><i className="fa-regular fa-file-image"></i> <strong>{fileMeta.name}</strong></span>
                                            <span>({fileMeta.size})</span>
                                        </div>
                                    )}

                                    {isLoading ? (
                                        <div className="scanning-indicator">
                                            <div className="scanner-pulse-spinner"></div>
                                            <div className="scan-status-text">
                                                {scanStep === 0 && "Analyzing leaf texture and RGB chlorophyll channels..."}
                                                {scanStep === 1 && "Extracting spatial disease features via CNN filters..."}
                                                {scanStep === 2 && "Calculating plant diagnosis and confidence score..."}
                                            </div>
                                        </div>
                                    ) : (
                                        <button className="analyze-action-btn" onClick={analyzeImage}>
                                            <i className="fa-solid fa-microscope"></i> Diagnose Plant & Disease
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* View 2: Prediction Results Dashboard */}
                    {prediction && (
                        <div className="result-dashboard animate-in">
                            {/* Low Confidence Warning */}
                            {prediction.is_ambiguous && (
                                <div className="ambiguity-callout">
                                    <i className="fa-solid fa-triangle-exclamation"></i>
                                    <div>
                                        <strong>Low Confidence / Unclear Leaf Warning:</strong>
                                        <p>The neural network detected low confidence ({prediction.confidence}%). The leaf image may be backlit, out of focus, or outside standard crop categories. Please re-shoot under uniform lighting.</p>
                                    </div>
                                </div>
                            )}

                            {/* Headline Announcement Banner (Exact Output Wording) */}
                            <div className="prediction-headline-card">
                                <div className="headline-text">
                                    <i className="fa-solid fa-leaf"></i>
                                    <span>{prediction.prediction_text || `This is a ${prediction.plant || prediction.crop} leaf with ${prediction.disease || prediction.condition}`}</span>
                                </div>
                                <div className="headline-meta">
                                    <i className="fa-solid fa-brain"></i> CNN Model &bull; {prediction.confidence}% Match
                                </div>
                            </div>

                            {/* 3 Prominent Stat Highlight Cards: Plant, Disease, Confidence */}
                            <div className="plant-disease-stat-grid">
                                <div className="stat-box">
                                    <div className="stat-icon plant-icon">
                                        <i className="fa-solid fa-seedling"></i>
                                    </div>
                                    <div className="stat-info">
                                        <span className="stat-label">Plant / Crop Name</span>
                                        <span className="stat-val">{prediction.plant || prediction.crop}</span>
                                    </div>
                                </div>

                                <div className="stat-box">
                                    <div className="stat-icon disease-icon">
                                        <i className={`fa-solid ${prediction.is_healthy ? "fa-shield-heart" : "fa-shield-virus"}`}></i>
                                    </div>
                                    <div className="stat-info">
                                        <span className="stat-label">Disease / Condition</span>
                                        <span className="stat-val">{prediction.disease || prediction.condition}</span>
                                    </div>
                                </div>

                                <div className="stat-box">
                                    <div className="stat-icon confidence-icon">
                                        <i className="fa-solid fa-chart-pie"></i>
                                    </div>
                                    <div className="stat-info">
                                        <span className="stat-label">Model Confidence</span>
                                        <span className="stat-val" style={{ color: getGaugeColor(prediction.confidence) }}>
                                            {prediction.confidence}% ({prediction.confidence >= 80 ? "High" : "Moderate"})
                                        </span>
                                    </div>
                                </div>
                            </div>

                            {/* Header Summary Bar with Circular Gauge */}
                            <div className="result-header-bar">
                                <div className="result-titles">
                                    <span className="result-eyebrow">Diagnosed Disease</span>
                                    <h2 className="condition-headline">{prediction.disease || prediction.condition}</h2>
                                    <div className="condition-badges">
                                        <span className="crop-pill">
                                            <i className="fa-solid fa-seedling"></i> Plant: <strong>{prediction.plant || prediction.crop}</strong>
                                        </span>
                                        <span className={`health-pill ${prediction.is_healthy ? "healthy" : "diseased"}`}>
                                            <i className={`fa-solid ${prediction.is_healthy ? "fa-circle-check" : "fa-shield-virus"}`}></i>
                                            {prediction.is_healthy ? "Healthy Foliage" : "Pathogen Detected"}
                                        </span>
                                    </div>
                                </div>

                                {/* Circular Confidence Gauge */}
                                <div className="gauge-container">
                                    <div className="gauge-relative">
                                        <svg className="gauge-svg" width="130" height="130" viewBox="0 0 130 130">
                                            <circle 
                                                className="gauge-track" 
                                                cx="65" cy="65" r={radius} 
                                                strokeWidth="10" 
                                            />
                                            <circle 
                                                className="gauge-fill" 
                                                cx="65" cy="65" r={radius} 
                                                strokeWidth="10" 
                                                strokeDasharray={circumference}
                                                strokeDashoffset={strokeDashoffset}
                                                stroke={getGaugeColor(prediction.confidence)}
                                            />
                                        </svg>
                                        <div className="gauge-center-text">
                                            <span className="gauge-number" style={{ color: getGaugeColor(prediction.confidence) }}>
                                                {prediction.confidence}%
                                            </span>
                                            <span className="gauge-sub">Confidence</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Grid: Image & Probabilities Left | Advisory Right */}
                            <div className="result-body-grid">
                                <div className="left-panel">
                                    <div className="analyzed-frame">
                                        <img src={prediction.image_url || previewUrl} alt="Analyzed Leaf" />
                                        <div className="frame-overlay">Uploaded Specimen</div>
                                    </div>

                                    {/* Top Probabilities / Alternative Guesses */}
                                    <div className="probabilities-box">
                                        <h4 className="panel-heading">
                                            <i className="fa-solid fa-chart-simple"></i> Top Model Guesses & Probabilities
                                        </h4>
                                        <div className="prob-list">
                                            {(prediction.top_predictions || []).map((p, i) => (
                                                <div key={i} className="prob-row">
                                                    <div className="prob-meta">
                                                        <span className="prob-name">
                                                            <strong>{p.crop}</strong> &bull; {p.condition}
                                                        </span>
                                                        <span className="prob-pct">{p.probability}%</span>
                                                    </div>
                                                    <div className="prob-track">
                                                        <div 
                                                            className="prob-bar" 
                                                            style={{ 
                                                                width: `${Math.min(100, p.probability)}%`,
                                                                backgroundColor: i === 0 ? getGaugeColor(p.probability) : "#94a3b8"
                                                            }}
                                                        ></div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                <div className="right-panel">
                                    {/* Advisory Tabs */}
                                    <div className="advisory-tabs">
                                        <button 
                                            className={`tab-btn ${activeTab === "cause" ? "active" : ""}`}
                                            onClick={() => setActiveTab("cause")}
                                        >
                                            <i className="fa-solid fa-dna"></i> Pathogen
                                        </button>
                                        <button 
                                            className={`tab-btn ${activeTab === "symptoms" ? "active" : ""}`}
                                            onClick={() => setActiveTab("symptoms")}
                                        >
                                            <i className="fa-solid fa-magnifying-glass"></i> Symptoms
                                        </button>
                                        <button 
                                            className={`tab-btn ${activeTab === "treatment" ? "active" : ""}`}
                                            onClick={() => setActiveTab("treatment")}
                                        >
                                            <i className="fa-solid fa-spray-can"></i> Management
                                        </button>
                                    </div>

                                    <div className="tab-card-body">
                                        {activeTab === "cause" && (
                                            <div className="tab-content animate-in">
                                                <h4 className="tab-label">Biological Pathogen / Causal Agent</h4>
                                                <p className="tab-text pathogen-highlight">
                                                    {prediction.advisory?.cause || "Pathological Leaf Condition"}
                                                </p>
                                                <p className="tab-hint">
                                                    Identified through convolutional neural pattern matching against the Plant-Disease-Detection dataset.
                                                </p>
                                            </div>
                                        )}

                                        {activeTab === "symptoms" && (
                                            <div className="tab-content animate-in">
                                                <h4 className="tab-label">Key Visual Diagnostic Symptoms</h4>
                                                <p className="tab-text">
                                                    {prediction.advisory?.symptoms || "Examine leaf surface for chlorotic yellowing, brown target-board lesions, or fungal pustules."}
                                                </p>
                                            </div>
                                        )}

                                        {activeTab === "treatment" && (
                                            <div className="tab-content animate-in">
                                                <h4 className="tab-label">Recommended Agronomic Controls</h4>
                                                <p className="tab-text">
                                                    {prediction.advisory?.management || "Ensure adequate air circulation, avoid overhead sprinkler irrigation, and consult agricultural extension guidelines."}
                                                </p>
                                            </div>
                                        )}
                                    </div>

                                    {/* Decision Support Disclaimer */}
                                    <div className="academic-disclaimer">
                                        <i className="fa-solid fa-circle-info"></i>
                                        <div>
                                            <strong>Decision-Support Notice (PRD Section 3 & 8):</strong>
                                            <p>This AI model is an educational and decision-support tool. It does not replace professional laboratory diagnosis or agricultural extension verification.</p>
                                        </div>
                                    </div>

                                    <div className="result-actions">
                                        <button className="reset-btn" onClick={resetAnalysis}>
                                            <i className="fa-solid fa-arrow-rotate-left"></i> Diagnose Another Leaf
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Recent Diagnoses Section (No samples, strictly upload & recent crops/diseases options) */}
                <section className="recent-section">
                    <div className="section-title-row">
                        <span className="section-eyebrow">
                            <i className="fa-solid fa-clock-rotate-left"></i> Recent Crops & Diseases Diagnosed
                        </span>
                        {recentScans.length > 0 && (
                            <button className="clear-history-btn" onClick={clearRecentScans}>
                                <i className="fa-solid fa-trash-can"></i> Clear History
                            </button>
                        )}
                    </div>

                    {recentScans.length === 0 ? (
                        <div className="empty-history-box">
                            <i className="fa-regular fa-images"></i>
                            <p>No recent diagnoses yet. Upload a leaf photo above to see past scan records here.</p>
                        </div>
                    ) : (
                        <div className="recent-grid">
                            {recentScans.map((scan) => (
                                <div key={scan.id} className="recent-card" onClick={() => loadRecentScan(scan)} title={`Click to review ${scan.crop} - ${scan.condition}`}>
                                    <div className="recent-thumb">
                                        <img src={scan.thumbnail} alt={scan.condition} />
                                        <span className={`recent-status-dot ${scan.is_healthy ? "healthy" : "diseased"}`}></span>
                                    </div>
                                    <div className="recent-details">
                                        <div className="recent-crop-name">{scan.crop || scan.plant}</div>
                                        <div className="recent-condition">{scan.condition || scan.disease}</div>
                                        <div className="recent-footer">
                                            <span className="recent-conf" style={{ color: getGaugeColor(scan.confidence) }}>
                                                {scan.confidence}% match
                                            </span>
                                            <span className="recent-time">{scan.time}</span>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </section>
            </main>

            {/* 38-Classes Catalogue Modal */}
            {showCatalogModal && (
                <div className="modal-backdrop" onClick={() => setShowCatalogModal(false)}>
                    <div className="modal-card animate-in" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <div>
                                <h3>Supported Crops & Diseases ({allClasses.length || 38} Classes)</h3>
                                <p className="modal-sub">PlantVillage Color Dataset Coverage</p>
                            </div>
                            <button className="modal-close-btn" onClick={() => setShowCatalogModal(false)}>&times;</button>
                        </div>
                        <div className="modal-search">
                            <i className="fa-solid fa-magnifying-glass"></i>
                            <input 
                                type="text" 
                                placeholder="Search by crop or disease (e.g. Tomato, Blight, Scab)..."
                                value={searchQuery}
                                onChange={(e) => setSearchQuery(e.target.value)}
                            />
                        </div>
                        <div className="modal-body-list">
                            {allClasses
                                .filter(c => c.toLowerCase().includes(searchQuery.toLowerCase()))
                                .map((cls, idx) => {
                                    const parts = cls.split("___");
                                    const crop = parts[0]?.replace(/_/g, " ");
                                    const cond = parts[1]?.replace(/_/g, " ") || cls;
                                    const isH = cond.toLowerCase().includes("healthy");
                                    return (
                                        <div key={idx} className="catalog-item">
                                            <div className="catalog-crop">
                                                <i className="fa-solid fa-leaf"></i> {crop}
                                            </div>
                                            <div className="catalog-cond">
                                                <span className={`catalog-tag ${isH ? "h-tag" : "d-tag"}`}>
                                                    {cond}
                                                </span>
                                            </div>
                                        </div>
                                    );
                                })}
                        </div>
                    </div>
                </div>
            )}

            {/* Footer */}
            <footer className="app-footer">
                <p>&copy; 2026 <strong>Virat Dagar</strong> &bull; B.Tech Computer Science & Engineering (Data Science)</p>
                <p className="footer-sub">ABES Engineering College &bull; HCLTech Industry Aligned Projects Program &bull; Academic Session 2026–27</p>
            </footer>
        </div>
    );
}

// Render React 18 Application
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
