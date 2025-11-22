"""
AI-Powered FMS Squat Analysis API - POC Version
File: api/index.py (Vercel deployment structure)
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
from datetime import datetime
from typing import Dict

# ============================================================
# CONFIG & SETUP
# ============================================================
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ============================================================
# FMS SQUAT ANALYSIS PROMPT
# ============================================================

FMS_SQUAT_PROMPT = """
You are an expert movement analyst specializing in Functional Movement Screening (FMS). 
Analyze this overhead deep squat video and provide a detailed assessment.

## Scoring Criteria

**3 (Optimal):**
- Thighs below parallel
- Torso and tibia parallel
- Heels flat, knees aligned over feet
- Arms/dowel overhead, no forward lean

**2 (Compensated):**
- Heels lift OR torso leans forward >10-15° OR mild valgus/varus at knees

**1 (Dysfunctional):**
- Cannot reach parallel depth, major loss of balance, or multiple compensations

**0 (Pain):**
- User reports pain during movement (this must be self-reported, not visible)

## Your Task

1. Observe the squat from start to finish
2. Identify specific movement patterns
3. Note any compensations or dysfunctions
4. Provide actionable feedback for improvement

## Required Response Format (JSON)

{
  "score": <0-3>,
  "classification": "<Optimal|Compensated|Dysfunctional|Pain>",
  "observations": {
    "depth": "<description of thigh position relative to parallel>",
    "torso": "<description of torso angle and forward lean>",
    "heels": "<flat or lifted>",
    "knees": "<alignment over feet, any valgus/varus>",
    "arms": "<overhead position maintained or not>"
  },
  "compensations_detected": ["<list of specific compensations observed>"],
  "strengths": ["<list of things done well>"],
  "improvements": ["<specific actionable suggestions for improvement>"],
  "mobility_focus_areas": ["<muscle groups or joints that may need mobility work>"],
  "summary": "<2-3 sentence overall assessment>"
}

CRITICAL RULES:
- Output ONLY valid JSON, no other text before or after
- Be specific and actionable in your feedback
- Base your assessment ONLY on what you can observe in the video
- If you cannot clearly see certain aspects, note that in your observations
- Do not assume pain - only score 0 if the user explicitly reports pain

Respond ONLY with the JSON object, nothing else.
"""

# ============================================================
# AI ANALYSIS FUNCTION
# ============================================================

def analyze_squat_video(video_base64: str, mime_type: str = "video/mp4", reported_pain: bool = False) -> Dict:
    """Analyze squat video using Gemini Vision API"""
    try:
        model = genai.GenerativeModel(MODEL)
        
        # Prepare the video data for Gemini
        video_part = {
            "mime_type": mime_type,
            "data": video_base64
        }
        
        # Modify prompt if user reported pain
        prompt = FMS_SQUAT_PROMPT
        if reported_pain:
            prompt += "\n\nIMPORTANT: The user has reported experiencing pain during this movement. Score should be 0 (Pain)."
        
        # Generate response with video
        response = model.generate_content([prompt, video_part])
        response_text = response.text.strip()
        
        # Clean up any markdown formatting if present
        if '```' in response_text:
            lines = response_text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if '```json' in line or '```' in line:
                    in_json = not in_json
                    continue
                if in_json or ('{' in line or '}' in line or '[' in line or ']' in line):
                    json_lines.append(line)
            response_text = '\n'.join(json_lines)
        
        # Parse the JSON response
        try:
            analysis_result = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["score", "classification", "observations", "summary"]
            for field in required_fields:
                if field not in analysis_result:
                    analysis_result[field] = None
            
            # Ensure score is valid
            if analysis_result.get("score") not in [0, 1, 2, 3]:
                analysis_result["score"] = None
                analysis_result["error"] = "Invalid score returned"
            
            return {
                "success": True,
                "analysis": analysis_result
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text}")
            return {
                "success": False,
                "error": "Failed to parse AI response",
                "raw_response": response_text
            }
            
    except Exception as e:
        print(f"Error in analyze_squat_video: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

# ============================================================
# FLASK APP
# ============================================================

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    @app.get("/health")
    def health():
        return jsonify({
            "ok": True,
            "model": MODEL,
            "service": "fms_squat_analysis_poc",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0-poc"
        })
    
    @app.post("/analyze")
    def analyze():
        """
        Main analysis endpoint - POC version (no user tracking)
        
        Expected JSON body:
        {
            "video": "<base64 encoded video>",
            "mime_type": "video/mp4",  // optional, defaults to video/mp4
            "reported_pain": false      // optional, if user reports pain
        }
        """
        try:
            data = request.get_json(force=True) or {}
            
            video_base64 = data.get("video")
            if not video_base64:
                return jsonify({"error": "video (base64) is required"}), 400
            
            mime_type = data.get("mime_type", "video/mp4")
            reported_pain = data.get("reported_pain", False)
            
            # Perform analysis
            result = analyze_squat_video(video_base64, mime_type, reported_pain)
            
            if result["success"]:
                return jsonify({
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "result": result["analysis"]
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result.get("error", "Analysis failed"),
                    "timestamp": datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            print(f"Error in analyze endpoint: {e}")
            import traceback
            traceback.print_exc()
            
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.post("/analyze-frame")
    def analyze_frame():
        """
        Analyze a single frame/image instead of video
        Useful for quick testing
        
        Expected JSON body:
        {
            "image": "<base64 encoded image>",
            "mime_type": "image/jpeg",  // optional
            "reported_pain": false       // optional
        }
        """
        try:
            data = request.get_json(force=True) or {}
            
            image_base64 = data.get("image")
            if not image_base64:
                return jsonify({"error": "image (base64) is required"}), 400
            
            mime_type = data.get("mime_type", "image/jpeg")
            reported_pain = data.get("reported_pain", False)
            
            # Use the same analysis function but with image
            result = analyze_squat_video(image_base64, mime_type, reported_pain)
            
            if result["success"]:
                return jsonify({
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "result": result["analysis"],
                    "note": "Analysis based on single frame - video analysis recommended for accuracy"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result.get("error", "Analysis failed"),
                    "timestamp": datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            print(f"Error in analyze-frame endpoint: {e}")
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    return app

# ============================================================
# VERCEL DEPLOYMENT
# ============================================================
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)