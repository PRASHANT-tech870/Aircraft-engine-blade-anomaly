# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import io
import base64
from groq import Groq
import sys
import os
from pydantic import BaseModel
from google import genai
import json
from fastapi.responses import FileResponse
from datetime import datetime

from typing import List
from fastapi.middleware.cors import CORSMiddleware

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Update imports to use correct paths
from MMR.models.MMR.MMR import MMR_base
from MMR.models.MMR.utils import ForwardHook, cal_anomaly_map
from MMR.config.defaults import get_cfg

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Initialize Groq client
client = Groq(api_key="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")

# Class labels
class_labels = ["ablation", "breakdown", "fracture", "groove", "good"]

# Add this function after initialization
def load_mmr_model(checkpoint_path, config_path, device):
    """Load and initialize the MMR model"""
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    
    mmr_model = MMR_base(cfg=cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    mmr_model.load_state_dict(checkpoint['mmr_base_state_dict'])
    mmr_model.to(device)
    mmr_model.eval()
    
    return mmr_model, cfg

# Update the initialize_models function with correct paths
def initialize_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize current model (WideResNet50)
    cur_model = models.wide_resnet50_2(pretrained=True)
    cur_model.to(device)
    cur_model.eval()
    
    # Initialize MMR model with correct paths
    checkpoint_path = os.path.join(current_dir, "MMR", "checkpoints", "aebad_S_AeBAD_S_MMR_model.pth")
    config_path = os.path.join(current_dir, "MMR", "method_config", "AeBAD_S", "MMR.yaml")
    
    mmr_model, cfg = load_mmr_model(checkpoint_path, config_path, device)
    
    return cur_model, mmr_model, cfg, device

# Update model initialization
cur_model, mmr_model, cfg, device = initialize_models()

# Replace the existing detect_anomaly function with this updated version
async def detect_anomaly(image_bytes: bytes, threshold=0.4):
    """Process image and detect anomalies"""
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Initialize teacher outputs dictionary
        teacher_outputs_dict = {}
        for extract_layer in cfg.TRAIN.MMR.layers_to_extract_from:
            forward_hook = ForwardHook(teacher_outputs_dict, extract_layer)
            network_layer = cur_model.__dict__["_modules"][extract_layer]
            network_layer[-1].register_forward_hook(forward_hook)
        
        # Define image transforms
        transform = transforms.Compose([
            transforms.Resize(cfg.DATASET.resize),
            transforms.CenterCrop(cfg.DATASET.imagesize),
            transforms.ToTensor(),
        ])
        
        # Process image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Get current model features
            teacher_outputs_dict.clear()
            _ = cur_model(image_tensor)
            multi_scale_features = [teacher_outputs_dict[key] 
                                  for key in cfg.TRAIN.MMR.layers_to_extract_from]
            
            # Get MMR features
            reverse_features = mmr_model(image_tensor, 
                                       mask_ratio=cfg.TRAIN.MMR.test_mask_ratio)
            multi_scale_reverse_features = [reverse_features[key] 
                                          for key in cfg.TRAIN.MMR.layers_to_extract_from]
            
            # Calculate anomaly map
            anomaly_map, _ = cal_anomaly_map(
                multi_scale_features, 
                multi_scale_reverse_features, 
                image_tensor.shape[-1],
                amap_mode='a'
            )
            
            # Apply Gaussian smoothing
            anomaly_map = gaussian_filter(anomaly_map[0], sigma=4)
            
            # Calculate final results
            anomaly_score = float(np.max(anomaly_map))
            is_anomaly = anomaly_score > threshold
            
            # Create visualization
            plt.figure(figsize=(12, 4), dpi=100)
            
            # Plot original image
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original Image')
            plt.axis('off')
            
            # Plot anomaly map
            plt.subplot(1, 2, 2)
            im = plt.imshow(anomaly_map, cmap='jet')
            plt.title(f"Anomaly Map\nScore: {anomaly_score:.3f}")
            plt.axis('off')
            plt.colorbar(im)
            
            plt.tight_layout()
            
            # Save plot to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode()
            plt.close()
            

            dict = {'success': True,
                'plot': plot_data,
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'anomaly_map': anomaly_map.tolist()}
            
        
            print(dict['anomaly_score'])
            return dict
            
    except Exception as e:
        return {'success': False, 'error': str(e)}

async def classify_image_type(image_bytes: bytes):
    """Classify the type of anomaly in the image"""
    try:
        # Initialize Groq client
        client = Groq(api_key="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")

        # Class labels
        class_labels = ["ablation", "breakdown", "fracture", "groove", "good"]

        # Convert image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Make API call to Groq with detailed prompt
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": (
                            "I am analyzing images of aircraft engine blades. Please classify the image into one of the following categories based on visible damage: "
                            "1. **Ablation**: Loss of material due to erosion or wear, usually represented by blackened or scorched areas, typically seen as small regions of surface damage."
                            "2. **Breakdown**: Holes or significant damage in the blade structure, indicating severe stress or material failure."
                            "3. **Fracture**: Loss or cutting out of part of the blade, usually seen as cracks or chunks missing from the edges."
                            "4. **Groove**: Shallow cuts or indentations along the surface of the blade, often caused by external factors or friction."
                            "5. **Good**: No visible damage, wear, or defects, with the blade in perfect working condition."
                            "Classify this aircraft engine blade image into exactly one of these categories: ablation, breakdown, fracture, groove, good. "
                            "Respond with only the category name in lowercase, nothing else."
                        )},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )

        # Get and validate prediction
        prediction = chat_completion.choices[0].message.content.lower().strip()
        if prediction not in class_labels:
            return {'success': False, 'error': 'Invalid classification'}
            
        return {'success': True, 'prediction': prediction}
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(..., max_size=10*1024*1024)):  # 10MB limit
    try:
        contents = await file.read()
        
        # Run both analyses in parallel
        anomaly_results = await detect_anomaly(contents)
        classification_results = await classify_image_type(contents)
        print("here")
        # Check for specific failures and provide detailed error messages
        if not anomaly_results['success']:
            raise HTTPException(
                status_code=400, 
                detail=f"Anomaly detection failed: {anomaly_results.get('error', 'Unknown error')}"
            )
        print(classification_results)   
        if not classification_results['success']:
            raise HTTPException(
                status_code=400, 
                detail=f"Classification failed: {classification_results.get('error', 'Unknown error')}"
            )
        
        return {
            'anomaly_results': anomaly_results,
            'classification_results': classification_results
        }
    except Exception as e:
        # Log the full error for debugging
        import traceback
        print(f"Error in analyze_image: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    

# Model to store sensor data and prediction

class SensorData(BaseModel):
    cycle: float  # Engine operational cycle number
    
    # Operating Settings
    operating_setting_1: float  # First operational parameter (OpSet1)
    operating_setting_2: float  # Second operational parameter (OpSet2)
    operating_setting_3: float  # Third operational parameter (OpSet3)
    
    # Temperature Sensors
    primary_temperature: float    # SensorMeasure1
    secondary_temperature: float  # SensorMeasure2
    tertiary_temperature: float   # SensorMeasure3
    quaternary_temperature: float # SensorMeasure4
    
    # Pressure Sensors
    primary_pressure: float    # SensorMeasure5
    secondary_pressure: float  # SensorMeasure6
    tertiary_pressure: float   # SensorMeasure7
    quaternary_pressure: float # SensorMeasure8
    
    # Speed/Rotation Sensors
    primary_speed: float    # SensorMeasure9
    secondary_speed: float  # SensorMeasure10
    tertiary_speed: float   # SensorMeasure11
    quaternary_speed: float # SensorMeasure12
    
    # Vibration/Mechanical Sensors
    primary_vibration: float   # SensorMeasure13
    secondary_vibration: float # SensorMeasure14
    
    # Flow Sensors
    primary_flow: float    # SensorMeasure15
    secondary_flow: float  # SensorMeasure16
    tertiary_flow: float   # SensorMeasure17
    
    # Additional Sensors
    pressure_ratio: float        # SensorMeasure18
    efficiency_indicator: float  # SensorMeasure19
    power_setting: float         # SensorMeasure20
    fuel_flow_rate: float        # SensorMeasure21
    
    prediction: str  # Condition classification (GOOD/MODERATE/BAD)


# Store the latest sensor data and prediction
latest_data = {}

@app.post("/update_sensor_data/")
async def update_sensor_data(sensor_data: SensorData):
    latest_data["data"] = sensor_data.dict()  # Save data
    return {"message": "Sensor data updated successfully"}

@app.get("/get_latest_data/")
async def get_latest_data():
    if latest_data:
        return latest_data
    return {"message": "No data available yet"}

# Initialize Gemini client after other initializations
genai_client = genai.Client(api_key="AIzaSyAzixKqoCq4RRJi5U4y11ndt7nVVigTNyw")

# Add this new endpoint
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import re
from datetime import datetime
from pathlib import Path

def create_pdf_report(content, filename):
    # Ensure reports directory exists
    Path("static/reports").mkdir(parents=True, exist_ok=True)
    
    # Setup document
    doc = SimpleDocTemplate(
        f"static/reports/{filename}",
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    # Styles
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name='CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#2c3e50')
    ))
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY
    ))
    styles.add(ParagraphStyle(
        name='CustomBold',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        fontName='Helvetica-Bold'
    ))

    # Story (content elements)
    story = []

    # Add title and timestamp
    story.append(Paragraph("Engine Health Report", styles['CustomTitle']))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Generated on: {timestamp}", styles['CustomBody']))
    story.append(Spacer(1, 20))

    # Markdown processing
    lines = content.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('# '):  # Level 1 Heading
            story.append(Paragraph(line[2:], styles['CustomHeading']))
            story.append(Spacer(1, 12))
        elif line.startswith('## '):  # Level 2 Heading
            story.append(Paragraph(line[3:], styles['Heading3']))
            story.append(Spacer(1, 10))
        elif re.match(r'^\*\*.*\*\*$', line):  # Bold Text
            story.append(Paragraph(line.replace('**', ''), styles['CustomBold']))
        elif line.startswith('* '):  # Unordered List
            list_item = ListFlowable([
                ListItem(Paragraph(line[2:], styles['CustomBody']), bulletColor=colors.black)
            ], bulletType='bullet')
            story.append(list_item)
        elif line:  # Regular Paragraph
            # Handle inline bold (e.g., **bold text**)
            line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            story.append(Paragraph(line, styles['CustomBody']))
        story.append(Spacer(1, 6))

    # Build PDF
    doc.build(story)
    return filename


@app.post("/generate_report")
async def generate_report(data: dict):
    try:
        # Format the data for the report
        anomaly_score = data.get('anomaly_score')
        blade_condition = data.get('blade_condition')
        sensor_data = data.get('sensor_data')
        
        # Create prompt for Gemini (modified to request markdown formatting)
        prompt = f"""
        Generate a professional aircraft engine health report based on the following data:
        
        1. Blade Analysis:
        - Anomaly Score: {anomaly_score}
        - Blade Condition: {blade_condition}
        
        2. Sensor Measurements:
        {json.dumps(sensor_data, indent=2)}
        
        Please format the report with the following markdown sections:
        
        # Executive Summary
        [Concise overview of findings]
        
        # Blade Condition Analysis
        [Detailed analysis of blade condition including anomaly score interpretation]
        
        # Sensor Data Analysis
        [Analysis of key sensor measurements and trends]
        
        # Health Status Assessment
        [Overall health status and risk assessment]
        
        # Recommendations
        [Specific maintenance and monitoring recommendations]
        
        Use professional technical language and include specific data points in your analysis.
        """
        
        # Generate report using Gemini
        response = genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        # Generate report filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"engine_health_report_{timestamp}.pdf"
        
        # Create PDF report
        create_pdf_report(response.text, filename)
            
        return {"success": True, "filename": filename}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/download_report/{filename}")
async def download_report(filename: str):
    return FileResponse(
        f"static/reports/{filename}",
        media_type='application/pdf',
        filename=filename
    )


@app.get("/training-results", response_class=HTMLResponse)
async def training_results(request: Request):
    return templates.TemplateResponse("results.html", {"request": request})