// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const resultsSection = document.getElementById('resultsSection');
const resultImage = document.getElementById('resultImage');
const detectionsList = document.getElementById('detectionsList');
const resultsStats = document.getElementById('resultsStats');
const loadingOverlay = document.getElementById('loadingOverlay');

// Navigation
function scrollToDemo() {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
}

function scrollToAbout() {
    document.getElementById('about').scrollIntoView({ behavior: 'smooth' });
}

// Active navigation link highlighting
window.addEventListener('scroll', () => {
    const sections = ['home', 'about', 'demo', 'results'];
    const navLinks = document.querySelectorAll('.nav-link');
    
    let current = '';
    sections.forEach(section => {
        const element = document.getElementById(section);
        const rect = element.getBoundingClientRect();
        if (rect.top <= 100 && rect.bottom >= 100) {
            current = section;
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// File Upload Handling
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

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
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// File Processing
function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPG, PNG)');
        return;
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }
    
    // Show loading
    showLoading();
    
    // Create FormData and send to API
    const formData = new FormData();
    formData.append('image', file);
    
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('An error occurred while processing the image: ' + error.message);
    });
}

// Display Results
function displayResults(data) {
    // Show results section
    resultsSection.style.display = 'block';
    
    // Display annotated image
    resultImage.src = data.annotated_image;
    
    // Update stats
    resultsStats.innerHTML = `
        <span><i class="fas fa-search"></i> ${data.total_detections} detections</span>
        <span><i class="fas fa-clock"></i> Processed in real-time</span>
    `;
    
    // Display detections list
    detectionsList.innerHTML = '';
    
    if (data.predictions && data.predictions.length > 0) {
        data.predictions.forEach((detection, index) => {
            const detectionItem = createDetectionItem(detection, index);
            detectionsList.appendChild(detectionItem);
        });
    } else {
        detectionsList.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 2rem;">No layout elements detected in this image.</p>';
    }
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Create Detection Item
function createDetectionItem(detection, index) {
    const item = document.createElement('div');
    item.className = 'detection-item';
    
    const confidence = Math.round(detection.score * 100);
    const bbox = detection.bbox;
    
    item.innerHTML = `
        <div class="detection-header">
            <span class="detection-class">${detection.category}</span>
            <span class="detection-confidence">${confidence}%</span>
        </div>
        <div class="detection-bbox">
            x: ${Math.round(bbox[0])}, y: ${Math.round(bbox[1])}, 
            w: ${Math.round(bbox[2])}, h: ${Math.round(bbox[3])}
        </div>
    `;
    
    // Add hover effect to highlight corresponding box in image
    item.addEventListener('mouseenter', () => {
        item.style.background = 'rgba(99, 102, 241, 0.1)';
        item.style.borderColor = 'var(--primary-color)';
    });
    
    item.addEventListener('mouseleave', () => {
        item.style.background = 'rgba(30, 41, 59, 0.5)';
        item.style.borderColor = 'var(--border-color)';
    });
    
    return item;
}

// Loading Functions
function showLoading() {
    loadingOverlay.classList.add('active');
}

function hideLoading() {
    loadingOverlay.classList.remove('active');
}

// Error Handling
function showError(message) {
    // Create error notification
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-notification';
    errorDiv.innerHTML = `
        <div class="error-content">
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    // Add error styles
    errorDiv.style.cssText = `
        position: fixed;
        top: 2rem;
        right: 2rem;
        background: rgba(239, 68, 68, 0.9);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        animation: slideIn 0.3s ease;
        max-width: 400px;
    `;
    
    errorDiv.querySelector('.error-content').style.cssText = `
        display: flex;
        align-items: center;
        gap: 0.75rem;
    `;
    
    errorDiv.querySelector('button').style.cssText = `
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        padding: 0.25rem;
        border-radius: 0.25rem;
        margin-left: auto;
    `;
    
    document.body.appendChild(errorDiv);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (errorDiv.parentElement) {
            errorDiv.remove();
        }
    }, 5000);
}

// Add CSS animation for error notification
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);

// Smooth scroll for navigation links
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        const targetId = link.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);
        if (targetElement) {
            targetElement.scrollIntoView({ behavior: 'smooth' });
        }
    });
});

// Add parallax effect to hero background elements
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const parallaxElements = document.querySelectorAll('.floating-element');
    
    parallaxElements.forEach((element, index) => {
        const speed = 0.5 + (index * 0.1);
        element.style.transform = `translateY(${scrolled * speed}px)`;
    });
});

// Initialize page
document.addEventListener('DOMContentLoaded', () => {
    // Check API health on page load
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            if (!data.model_loaded) {
                console.warn('Model not loaded. Using fallback model for demo.');
            }
        })
        .catch(error => {
            console.error('API health check failed:', error);
        });
    
    // Add loading animation to feature cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.animation = 'fadeInUp 0.6s ease forwards';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    document.querySelectorAll('.feature-card, .metric-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        observer.observe(card);
    });
});

// Add fadeInUp animation
const fadeInUpStyle = document.createElement('style');
fadeInUpStyle.textContent = `
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(fadeInUpStyle);

