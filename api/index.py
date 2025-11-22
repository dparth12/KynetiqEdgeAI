"""
AI-Powered FMS Squat Analysis API - POC Version
File: api/index.py (Vercel deployment structure)

Updated with improved prompt for skeleton overlay detection and camera angle flexibility
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
You are an expert movement analyst and physical therapist specializing in Functional Movement Screening (FMS). 
Analyze this overhead deep squat video and provide a detailed assessment.

## Video Context - IMPORTANT
- This video includes a POSE DETECTION SKELETON OVERLAY showing joint positions with colored lines/markers
- The skeleton overlay highlights: shoulders, elbows, wrists, hips, knees, and ankles
- USE THE SKELETON MARKERS to accurately assess joint angles, alignment, and movement quality
- The person is likely positioned in a SIDE/PROFILE VIEW for optimal squat assessment
- If the view is frontal or at an angle, adapt your assessment accordingly and note any limitations

## How to Use the Skeleton Overlay
- **Knee angle**: Look at the angle formed by hip-knee-ankle markers
- **Torso angle**: Compare the line from hip to shoulder against vertical
- **Depth assessment**: Check if hip marker drops below knee marker level
- **Knee tracking**: In frontal view, check if knee markers stay aligned over ankle markers
- **Arm position**: Check shoulder-elbow-wrist alignment for overhead position

## FMS Scoring Criteria

**Score 3 (Optimal):**
- Thighs reach below parallel (hip crease below knee)
- Torso and tibia remain parallel (minimal forward lean)
- Heels stay flat on the ground throughout movement
- Knees track over feet with no valgus (inward) or varus (outward) collapse
- Arms/dowel maintained overhead without excessive forward lean
- Smooth, controlled movement throughout

**Score 2 (Compensated):**
- Achieves depth but with ONE OR MORE compensations:
  - Heels lift off the ground
  - Torso leans forward >15-20° from vertical
  - Mild knee valgus or varus (knees collapse in or bow out)
  - Arms drop forward from overhead position
- Movement is still functional but shows mobility/stability limitations

**Score 1 (Dysfunctional):**
- Cannot reach parallel depth even with compensations
- Major loss of balance during movement
- Multiple significant compensations occurring together
- Unable to keep arms overhead
- Movement pattern is broken or unsafe

**Score 0 (Pain):**
- User explicitly reports pain during the movement
- NOTE: Only score 0 if pain is self-reported, NOT based on visual observation

## Your Analysis Task

1. **Observe** the full squat movement from start to finish
2. **Use the skeleton overlay** to measure/estimate joint angles
3. **Identify** specific movement patterns and compensations
4. **Assess** against the FMS criteria above
5. **Provide** actionable feedback for improvement

## Required Response Format (JSON)

{
  "score": <0-3>,
  "classification": "<Optimal|Compensated|Dysfunctional|Pain>",
  "observations": {
    "depth": "<specific description of thigh/hip position relative to parallel, reference skeleton markers if visible>",
    "torso": "<description of torso angle, estimate degrees of forward lean if possible>",
    "heels": "<flat throughout, lift at bottom, or unable to assess from this angle>",
    "knees": "<alignment over feet, any valgus/varus, reference knee markers>",
    "arms": "<overhead position maintained, dropped forward, or not applicable>"
  },
  "compensations_detected": [
    "<specific compensation 1>",
    "<specific compensation 2>"
  ],
  "strengths": [
    "<specific thing done well 1>",
    "<specific thing done well 2>"
  ],
  "improvements": [
    "<specific, actionable suggestion 1>",
    "<specific, actionable suggestion 2>",
    "<specific, actionable suggestion 3>"
  ],
  "mobility_focus_areas": [
    "<muscle group or joint that needs mobility work>",
    "<another area if applicable>"
  ],
  "summary": "<2-3 sentence overall assessment explaining the score and primary recommendations>"
}

## CRITICAL RULES
1. Output ONLY valid JSON - no text before or after the JSON object
2. Be SPECIFIC and reference what you see in the skeleton overlay
3. Provide ACTIONABLE feedback that the user can implement
4. Base assessment ONLY on what is visible - note if view limits certain observations
5. Do NOT assume pain - only score 0 if explicitly told user reported pain
6. If skeleton overlay is not visible or unclear, base assessment on body position
7. Be encouraging while being honest about areas for improvement

Respond with ONLY the JSON object.
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
            prompt += "\n\n⚠️ USER REPORTED PAIN: The user has indicated they experienced pain during this movement. The score MUST be 0 (Pain) regardless of movement quality observed."
        
        # Generate response with video
        response = model.generate_content([prompt, video_part])
        response_text = response.text.strip()
        
        # Clean up any markdown formatting if present
        if '```' in response_text:
            lines = response_text.split('\n')
            json_lines = []
            in_json = False
            for line in lines:
                if '```json' in line:
                    in_json = True
                    continue
                elif '```' in line and in_json:
                    in_json = False
                    continue
                if in_json:
                    json_lines.append(line)
            if json_lines:
                response_text = '\n'.join(json_lines)
        
        # Try to extract JSON if there's extra text
        if not response_text.startswith('{'):
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx + 1]
        
        # Parse the JSON response
        try:
            analysis_result = json.loads(response_text)
            
            # Validate and ensure all required fields exist
            required_fields = {
                "score": None,
                "classification": "Unknown",
                "observations": {
                    "depth": "Not assessed",
                    "torso": "Not assessed",
                    "heels": "Not assessed",
                    "knees": "Not assessed",
                    "arms": "Not assessed"
                },
                "compensations_detected": [],
                "strengths": [],
                "improvements": [],
                "mobility_focus_areas": [],
                "summary": "Analysis completed"
            }
            
            # Fill in missing fields with defaults
            for field, default in required_fields.items():
                if field not in analysis_result:
                    analysis_result[field] = default
                elif field == "observations" and isinstance(default, dict):
                    for obs_field, obs_default in default.items():
                        if obs_field not in analysis_result[field]:
                            analysis_result[field][obs_field] = obs_default
            
            # Validate score
            if analysis_result.get("score") not in [0, 1, 2, 3]:
                # Try to infer from classification
                classification_map = {
                    "Optimal": 3,
                    "Compensated": 2,
                    "Dysfunctional": 1,
                    "Pain": 0
                }
                analysis_result["score"] = classification_map.get(
                    analysis_result.get("classification"), 
                    2  # Default to compensated if unknown
                )
            
            # Ensure classification matches score
            score_classification_map = {
                3: "Optimal",
                2: "Compensated", 
                1: "Dysfunctional",
                0: "Pain"
            }
            if analysis_result.get("classification") not in score_classification_map.values():
                analysis_result["classification"] = score_classification_map.get(
                    analysis_result["score"], 
                    "Compensated"
                )
            
            return {
                "success": True,
                "analysis": analysis_result
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response text: {response_text[:500]}...")
            return {
                "success": False,
                "error": "Failed to parse AI response as JSON",
                "raw_response": response_text[:1000]
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
            "version": "1.1-poc",
            "features": [
                "skeleton_overlay_detection",
                "multi_angle_support",
                "pain_reporting"
            ]
        })
    
    @app.post("/analyze")
    def analyze():
        """
        Main video analysis endpoint
        
        Expected JSON body:
        {
            "video": "<base64 encoded video>",
            "mime_type": "video/mp4",  // optional, defaults to video/mp4
            "reported_pain": false      // optional, if user reports pain during movement
        }
        
        Returns:
        {
            "success": true,
            "timestamp": "...",
            "result": {
                "score": 0-3,
                "classification": "Optimal|Compensated|Dysfunctional|Pain",
                "observations": {...},
                "compensations_detected": [...],
                "strengths": [...],
                "improvements": [...],
                "mobility_focus_areas": [...],
                "summary": "..."
            }
        }
        """
        try:
            data = request.get_json(force=True) or {}
            
            video_base64 = data.get("video")
            if not video_base64:
                return jsonify({
                    "success": False,
                    "error": "video (base64) is required",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            mime_type = data.get("mime_type", "video/mp4")
            reported_pain = data.get("reported_pain", False)
            
            # Log request info (without the actual video data)
            print(f"Analyze request - mime_type: {mime_type}, reported_pain: {reported_pain}, video_length: {len(video_base64)}")
            
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
                    "raw_response": result.get("raw_response"),
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
        Useful for quick testing or snapshot analysis
        
        Expected JSON body:
        {
            "image": "<base64 encoded image>",
            "mime_type": "image/jpeg",  // optional, defaults to image/jpeg
            "reported_pain": false       // optional
        }
        """
        try:
            data = request.get_json(force=True) or {}
            
            image_base64 = data.get("image")
            if not image_base64:
                return jsonify({
                    "success": False,
                    "error": "image (base64) is required",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            mime_type = data.get("mime_type", "image/jpeg")
            reported_pain = data.get("reported_pain", False)
            
            print(f"Analyze-frame request - mime_type: {mime_type}, image_length: {len(image_base64)}")
            
            # Use the same analysis function but with image
            result = analyze_squat_video(image_base64, mime_type, reported_pain)
            
            if result["success"]:
                return jsonify({
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "result": result["analysis"],
                    "note": "Analysis based on single frame - video analysis recommended for more accurate assessment"
                })
            else:
                return jsonify({
                    "success": False,
                    "error": result.get("error", "Analysis failed"),
                    "timestamp": datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            print(f"Error in analyze-frame endpoint: {e}")
            import traceback
            traceback.print_exc()
            
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.get("/")
    def root():
        """Root endpoint with API info"""
        return jsonify({
            "service": "KynetiqEdge FMS Squat Analysis API",
            "version": "1.1-poc",
            "endpoints": {
                "GET /health": "Health check and service info",
                "POST /analyze": "Analyze squat video (base64)",
                "POST /analyze-frame": "Analyze single image frame (base64)"
            },
            "documentation": {
                "video_format": "MP4 recommended, base64 encoded",
                "skeleton_overlay": "Videos with pose detection skeleton overlay are supported and recommended",
                "camera_angle": "Side/profile view recommended for best results"
            }
        })
    
    return app

# ============================================================
# VERCEL DEPLOYMENT
# ============================================================
app = create_app()

if __name__ == "__main__":
    print("Starting FMS Squat Analysis API...")
    print(f"Model: {MODEL}")
    app.run(host="0.0.0.0", port=8000, debug=True)