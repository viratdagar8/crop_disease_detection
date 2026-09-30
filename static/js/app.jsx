const { useState, useEffect, useRef } = React;

// Sample quick-test images available in the project
const SAMPLE_LEAVES = [
    {
        id: "tomato",
        title: "Tomato Early Blight",
        crop: "Tomato",
        expected: "Early Blight",
        file: "/static/sample_images/tomato_early_blight.jpg",
        icon: "fa-seedling",
        tag: "Fungal Lesions"
    },
    {
        id: "potato",
        title: "Potato Late Blight",
        crop: "Potato",
        expected: "Late Blight",
        file: "/static/sample_images/potato_late_blight.jpg",
        icon: "fa-cubes-stacked",
        tag: "Oomycete Rot"
    },
    {
        id: "corn",
        title: "Corn Common Rust",
        crop: "Corn (Maize)",
        expected: "Common Rust",
        file: "/static/sample_images/corn_common_rust.jpg",
        icon: "fa-wheat-awn",
        tag: "Rust Pustules"
    },
    {
        id: "apple",
        title: "Apple Healthy Foliage",
        crop: "Apple",
        expected: "Healthy",
        file: "/static/sample_images/apple_healthy.jpg",
        icon: "fa-apple-whole",
        tag: "Healthy Leaf"
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
            }, 700);
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

    const handleQuickSelect = async (sample) => {
        setErrorMsg("");
        setPrediction(null);
        try {
            const res = await fetch(sample.file);
            const blob = await res.blob();
            const file = new File([blob], `${sample.id}_sample.jpg`, { type: "image/jpeg" });
            handleFile(file);
        } catch (err) {
            console.error("Error loading sample:", err);
            setErrorMsg("Could not load sample image.");
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
        } catch (err) {
            setErrorMsg(err.message || "Failed to analyze leaf image.");
        } finally {
            setIsLoading(false);
        }
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
                            <div className="brand-tagline">Deep Learning Vision System &bull; React 18 Edition</div>
                        </div>
                    </div>

                    <div className="nav-meta">
                        <button className="catalog-btn" onClick={() => setShowCatalogModal(true)}>
                            <i className="fa-solid fa-book-open"></i> Supported Crops (38 Classes)
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
                        Intelligent Crop Disease Classification with <span className="highlight-text">Convolutional Neural Networks</span>
                    </h1>
                    <p className="hero-subtext">
                        Upload or test a high-resolution crop leaf photograph. Our deep learning vision pipeline pre-processes the tensor, extracts spatial lesion feature maps, and computes disease probabilities with agronomic management advice.
                    </p>
                </section>

                {/* 1-Click Quick Test Cards */}
                <section className="quick-test-section">
                    <div className="section-title-row">
                        <span className="section-eyebrow"><i className="fa-solid fa-bolt"></i> 1-Click Test Samples</span>
                        <span className="section-hint">Click any leaf to test the system immediately</span>
                    </div>
                    <div className="sample-grid">
                        {SAMPLE_LEAVES.map((sample) => (
                            <div 
                                key={sample.id} 
                                className="sample-card" 
                                onClick={() => handleQuickSelect(sample)}
                            >
                                <div className="sample-img-wrap">
                                    <img src={sample.file} alt={sample.title} />
                                    <span className="sample-badge">{sample.tag}</span>
                                </div>
                                <div className="sample-info">
                                    <div className="sample-crop"><i className={`fa-solid ${sample.icon}`}></i> {sample.crop}</div>
                                    <div className="sample-title">{sample.expected}</div>
                                </div>
                            </div>
                        ))}
                    </div>
                </section>

                {/* Alert Box */}
                {errorMsg && (
                    <div className="alert-banner">
                        <i className="fa-solid fa-circle-exclamation"></i>
                        <span>{errorMsg}</span>
                        <button className="alert-dismiss" onClick={() => setErrorMsg("")}>&times;</button>
                    </div>
                )}

                {/* Main Workspace */}
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
                                    <h3 className="dropzone-heading">Drag & drop your leaf image here</h3>
                                    <p className="dropzone-sub">or <span className="highlight-link">browse files</span> from your computer</p>
                                    <div className="dropzone-specs">
                                        <span><i className="fa-regular fa-image"></i> JPG, PNG, WEBP</span>
                                        <span>&bull;</span>
                                        <span><i className="fa-solid fa-weight-hanging"></i> Up to 16 MB</span>
                                        <span>&bull;</span>
                                        <span><i className="fa-solid fa-expand"></i> Auto 128×128 Normalized</span>
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
                                                {scanStep === 0 && "Normalizing RGB channels & converting tensor..."}
                                                {scanStep === 1 && "Running 4-block Conv2D feature extraction..."}
                                                {scanStep === 2 && "Computing Softmax probabilities & agronomic advisory..."}
                                            </div>
                                        </div>
                                    ) : (
                                        <button className="analyze-action-btn" onClick={analyzeImage}>
                                            <i className="fa-solid fa-microscope"></i> Run Neural Network Analysis
                                        </button>
                                    )}
                                </div>
                            )}
                        </div>
                    )}

                    {/* View 2: Prediction Results Dashboard */}
                    {prediction && (
                        <div className="result-dashboard animate-in">
                            {/* Low Confidence Warning (Solves lagging confidence ambiguity) */}
                            {prediction.is_ambiguous && (
                                <div className="ambiguity-callout">
                                    <i className="fa-solid fa-triangle-exclamation"></i>
                                    <div>
                                        <strong>Low Confidence / Ambiguity Warning:</strong>
                                        <p>The neural network detected low confidence ({prediction.confidence}%). The leaf image may be backlit, out of focus, or outside the 14 supported crop categories. Please inspect under clear, uniform lighting.</p>
                                    </div>
                                </div>
                            )}

                            {/* Header Summary */}
                            <div className="result-header-bar">
                                <div className="result-titles">
                                    <span className="result-eyebrow">Diagnosed Condition</span>
                                    <h2 className="condition-headline">{prediction.condition}</h2>
                                    <div className="condition-badges">
                                        <span className="crop-pill">
                                            <i className="fa-solid fa-seedling"></i> {prediction.crop}
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
                                        <div className="frame-overlay">Input Image</div>
                                    </div>

                                    {/* Top Probabilities */}
                                    <div className="probabilities-box">
                                        <h4 className="panel-heading">
                                            <i className="fa-solid fa-chart-simple"></i> Top Model Probabilities
                                        </h4>
                                        <div className="prob-list">
                                            {prediction.top_predictions.map((p, i) => (
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
                                                                width: `${p.probability}%`,
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
                                                    Classification identified through spatial convolution pattern matching against PlantVillage RGB leaf archives.
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
                                                <h4 className="tab-label">Recommended Agronomic & Cultural Controls</h4>
                                                <p className="tab-text">
                                                    {prediction.advisory?.management || "Ensure adequate air circulation, avoid overhead sprinkler irrigation, and consult local extension guidelines."}
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
                                            <i className="fa-solid fa-arrow-rotate-left"></i> Analyze Another Leaf
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </main>

            {/* 38-Classes Catalogue Modal */}
            {showCatalogModal && (
                <div className="modal-backdrop" onClick={() => setShowCatalogModal(false)}>
                    <div className="modal-card animate-in" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <div>
                                <h3>Supported Crops & Diseases ({allClasses.length} Classes)</h3>
                                <p className="modal-sub">PlantVillage Color Dataset Architecture Coverage</p>
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
