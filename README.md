# Aircraft Engine health Monitoring System

A comprehensive system for monitoring and analyzing turbine performance using AI-powered image analysis and sensor data processing.

## Setup

### API Keys Required
```
GROQ_API_KEY     # For Groq model integration
GOOGLE_API_KEY   # For Gemini AI integration
```

### Running the Application

#### Backend (FastAPI)
```bash
cd /path/to/project
uvicorn app:app --reload
```
Backend runs on: `http://localhost:8000`

#### Frontend (Streamlit)
```bash
cd /path/to/project
streamlit run engine.py
```
Frontend will open automatically in your browser.

## API Endpoints

### Image Analysis
- **POST** `/analyze`
  - Analyzes blade images for defects
  - Supports image upload up to 10MB
  - Returns anomaly detection and classification results

### Sensor Data Management
- **POST** `/update_sensor_data/`
- **GET** `/get_latest_data/`
  - Handles real-time sensor data updates
  - Returns latest sensor readings and predictions

### Report Generation
- **POST** `/generate_report`
- **GET** `/download_report/{filename}`
  - Creates detailed PDF reports with AI analysis
  - Supports report download functionality

## Sensor Data Structure

The system monitors 21 different sensor measurements:

### Operational Parameters
- Operational Settings (3 parameters)
- Temperature Readings (4 sensors)
- Pressure Readings (4 sensors)
- Speed/Rotation Measurements (4 sensors)
- Vibration Readings (2 sensors)
- Flow Measurements (3 sensors)

### Additional Parameters
- Pressure Ratio
- Efficiency Indicator
- Power Setting
- Fuel Flow Rate

## Using the Streamlit Interface

### Data Input Methods

#### Manual Input
- Individual sensor value entry
- Real-time validation

#### Bulk Input
- Paste comma-separated values
- Automatic validation and mapping

### Predefined Scenarios
- GOOD condition preset
- MODERATE condition preset
- VERY BAD condition preset

### Getting Predictions
1. Enter/paste sensor values
2. Click "Submit"
3. View prediction results
4. Data automatically syncs with backend

## Error Handling

The system includes comprehensive error handling for:
- Invalid sensor data input
- Network communication issues
- Image processing errors
- API integration failures
- File handling problems

## System Requirements

- Python 3.8+
- Internet connection for API services
- Sufficient RAM for model loading (4GB minimum)
- Storage space for report generation

## Directory Structure

```
project/
├── app.py            # FastAPI backend
├── engine.py         # Streamlit frontend
├── static/           # Static files
│   └── reports/      # Generated PDF reports
├── templates/        # HTML templates
└── MMR/              # Model files
    └── checkpoints/  # Saved model states
```

## Notes

- Keep both FastAPI and Streamlit applications running simultaneously
- Monitor console output for debugging information
- Regular maintenance of reports directory recommended
- Backup your API keys securely

## Troubleshooting

### Connection Issues
- Verify both services are running
- Check port availability
- Confirm API keys are valid

### Model Loading Errors
- Ensure sufficient system memory
- Verify model file integrity
- Check Python environment

### Report Generation Issues
- Verify write permissions
- Check disk space
- Monitor API response times

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
