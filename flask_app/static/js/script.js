// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const uploadSection = document.getElementById('uploadSection');
const loadingState = document.getElementById('loadingState');
const resultsSection = document.getElementById('resultsSection');
const analyzeAnotherBtn = document.getElementById('analyzeAnotherBtn');

// Upload button click
uploadBtn.addEventListener('click', () => {
    fileInput.click();
});

// Upload area click
uploadArea.addEventListener('click', (e) => {
    if (e.target !== uploadBtn) {
        fileInput.click();
    }
});

// File input change
fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
    }
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// Handle file upload
function handleFile(file) {
    // Validate file type
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        alert('Please upload a valid image file (PNG, JPG, JPEG, BMP, or TIFF)');
        return;
    }

    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }

    // Show loading state
    uploadSection.style.display = 'none';
    loadingState.style.display = 'block';
    resultsSection.style.display = 'none';

    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    // Send to backend
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        displayResults(data);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error analyzing image: ' + error.message);
        resetUpload();
    });
}

// Display results
function displayResults(data) {
    // Hide loading, show results
    loadingState.style.display = 'none';
    resultsSection.style.display = 'block';

    // Display uploaded image
    document.getElementById('uploadedImage').src = data.image;

    // Display prediction
    const predictionBadge = document.getElementById('predictionBadge');
    const predictionText = document.getElementById('predictionText');
    
    predictionText.textContent = data.class_name;
    
    if (data.predicted_class === 1) {
        predictionBadge.className = 'prediction-badge positive';
    } else {
        predictionBadge.className = 'prediction-badge negative';
    }

    // Display confidence
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceValue = document.getElementById('confidenceValue');
    
    confidenceFill.style.width = data.confidence + '%';
    confidenceValue.textContent = data.confidence.toFixed(2) + '%';

    // Display detailed probabilities
    document.getElementById('probPositive').textContent = data.probability_positive.toFixed(2) + '%';
    document.getElementById('probNegative').textContent = data.probability_negative.toFixed(2) + '%';
    
    document.getElementById('fillPositive').style.width = data.probability_positive + '%';
    document.getElementById('fillNegative').style.width = data.probability_negative + '%';

    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Analyze another image
analyzeAnotherBtn.addEventListener('click', resetUpload);

function resetUpload() {
    uploadSection.style.display = 'block';
    loadingState.style.display = 'none';
    resultsSection.style.display = 'none';
    fileInput.value = '';
    uploadArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Prevent default drag behavior on document
document.addEventListener('dragover', (e) => e.preventDefault());
document.addEventListener('drop', (e) => e.preventDefault());