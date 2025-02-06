// static/script.js
document.addEventListener('DOMContentLoaded', function() {
    // Elements for image analysis
    const dropZone = document.getElementById('dropZone');
    const uploadBtn = document.getElementById('uploadBtn');
    const imageInput = document.getElementById('imageInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const fileName = document.getElementById('fileName');
    const resultsSection = document.getElementById('resultsSection');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const originalImage = document.getElementById('originalImage');
    const anomalyMap = document.getElementById('anomalyMap');
    const anomalyScore = document.getElementById('anomalyScore');
    const classification = document.getElementById('classification');

    // Elements for sensor data
    const fetchSensorBtn = document.getElementById('fetchSensorBtn');
    const sensorDataGrid = document.getElementById('sensorDataGrid');

    // Elements for report generation
    const generateReportBtn = document.getElementById('generateReportBtn');
    const reportStatus = document.getElementById('reportStatus');

    // Store latest sensor data
    let latest_data = null;

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            imageInput.files = files;
            updateFileInfo(files[0]);
        }
    }

    // Click upload
    uploadBtn.addEventListener('click', () => {
        imageInput.click();
    });

    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            updateFileInfo(e.target.files[0]);
        }
    });

    function validateFile(file) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        if (file.size > maxSize) {
            throw new Error('File size too large. Maximum size is 10MB');
        }

        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!allowedTypes.includes(file.type)) {
            throw new Error('Invalid file type. Please upload a JPEG or PNG image');
        }

        return true;
    }

    function updateFileInfo(file) {
        try {
            validateFile(file);
            fileName.textContent = file.name;
            analyzeBtn.disabled = false;
            
            const reader = new FileReader();
            reader.onload = (e) => {
                originalImage.src = e.target.result;
            };
            reader.readAsDataURL(file);
        } catch (error) {
            showError(error.message);
            fileName.textContent = '';
            analyzeBtn.disabled = true;
        }
    }

    // Image Analysis
    analyzeBtn.addEventListener('click', async () => {
        if (!imageInput.files.length) return;

        loadingSpinner.style.display = 'flex';
        resultsSection.style.display = 'none';

        const formData = new FormData();
        formData.append('file', imageInput.files[0]);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Analysis failed');
            }

            const data = await response.json();
            
            if (!data.anomaly_results || !data.classification_results) {
                throw new Error('Invalid response format from server');
            }
            
            anomalyMap.src = `data:image/png;base64,${data.anomaly_results.plot}`;
            anomalyScore.textContent = data.anomaly_results.anomaly_score.toFixed(3);
            classification.textContent = data.classification_results.prediction;
            
            resultsSection.style.display = 'block';
        } catch (error) {
            console.error('Analysis error:', error);
            showError(`Error analyzing image: ${error.message}`);
        } finally {
            loadingSpinner.style.display = 'none';
        }
    });

    // Sensor Data Fetching
    fetchSensorBtn.addEventListener('click', async () => {
        try {
            fetchSensorBtn.disabled = true;
            fetchSensorBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';

            const response = await fetch('/get_latest_data');
            const data = await response.json();

            if (data.data) {
                latest_data = data; // Store the latest data
                sensorDataGrid.innerHTML = '';

                Object.entries(data.data).forEach(([key, value]) => {
                    const parameterCard = document.createElement('div');
                    parameterCard.className = 'sensor-parameter';
                    
                    const displayName = key.replace(/_/g, ' ')
                        .split(' ')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ');

                    if (key === 'prediction') {
                        parameterCard.innerHTML = `
                            <h4>${displayName}</h4>
                            <span class="prediction-value">${value}</span>
                        `;
                    } else {
                        parameterCard.innerHTML = `
                            <h4>${displayName}</h4>
                            <span class="value">${typeof value === 'number' ? value.toFixed(2) : value}</span>
                        `;
                    }
                    
                    sensorDataGrid.appendChild(parameterCard);
                });
            } else {
                showError('No sensor data available');
            }
        } catch (error) {
            console.error('Error fetching sensor data:', error);
            showError('Error fetching sensor data');
        } finally {
            fetchSensorBtn.disabled = false;
            fetchSensorBtn.innerHTML = '<i class="fas fa-sync"></i> Fetch Sensor Data';
        }
    });

    // Report Generation
    async function generateReport(anomalyScore, bladeCondition, sensorData) {
        try {
            generateReportBtn.disabled = true;
            generateReportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Report...';
            reportStatus.innerHTML = 'Generating report, please wait...';

            const response = await fetch('/generate_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    anomaly_score: anomalyScore,
                    blade_condition: bladeCondition,
                    sensor_data: sensorData
                })
            });

            const data = await response.json();

            if (data.success) {
                reportStatus.innerHTML = `
                    <div class="success-message">Report generated successfully!</div>
                    <a href="/download_report/${data.filename}" class="download-link" target="_blank">
                        <i class="fas fa-download"></i> Download Report
                    </a>
                `;
            } else {
                throw new Error(data.error || 'Failed to generate report');
            }
        } catch (error) {
            console.error('Report generation error:', error);
            showError('Error generating report: ' + error.message);
            reportStatus.innerHTML = 'Failed to generate report. Please try again.';
        } finally {
            generateReportBtn.disabled = false;
            generateReportBtn.innerHTML = '<i class="fas fa-file-medical"></i> Generate Report';
        }
    }

    generateReportBtn.addEventListener('click', async () => {
        if (!anomalyScore.textContent || !classification.textContent || !latest_data) {
            showError('Please complete blade analysis and fetch sensor data first');
            return;
        }

        await generateReport(
            parseFloat(anomalyScore.textContent),
            classification.textContent,
            latest_data.data
        );
    });

    function showError(message) {
        const toast = document.createElement('div');
        toast.className = 'toast error';
        toast.innerHTML = `
            <i class="fas fa-exclamation-circle"></i>
            <span>${message}</span>
        `;
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.remove();
        }, 5000);
    }
});