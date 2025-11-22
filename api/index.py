"""
AI-Powered FMS Squat Analysis API - POC Version
File: api/index.py (Vercel deployment structure)

Updated to use pose detection metadata from the app alongside Gemini's visual analysis.
The AI focuses on what pose detection CAN'T capture (torso lean, heel lift, knee valgus, 
arm position, balance) while trusting the app's depth/knee angle data.
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
from datetime import datetime
from typing import Dict, Optional

# ============================================================
# CONFIG & SETUP
# ============================================================
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# ============================================================
# FMS SQUAT ANALYSIS PROMPT (Dynamic based on pose data)
# ============================================================

def build_analysis_prompt(pose_data: Optional[Dict] = None) -> str:
    """Build the analysis prompt, incorporating pose detection data if available."""
    
    # Base context about what the AI should focus on
    base_prompt = """
You are an expert movement analyst and physical therapist specializing in Functional Movement Screening (FMS). 
Analyze this overhead deep squat video and provide a detailed assessment.

## Your Role
You are working alongside a pose detection system (QuickPose SDK) that tracks joint positions in real-time.
The pose detection system provides ACCURATE data for:
- Knee angle measurements
- Whether squat depth was reached (below parallel)
- Number of squats performed

Your job is to analyze what the pose detection CANNOT reliably capture:
1. **Torso position** - Forward lean angle, uprightness
2. **Heel contact** - Whether heels lift off the ground
3. **Knee tracking** - Valgus (inward collapse) or varus (outward bow)
4. **Arm/overhead position** - If arms are maintained overhead
5. **Balance & control** - Stability throughout movement
6. **Overall movement quality** - Smoothness, hesitation, compensation patterns

## Video Context
- The video includes a SKELETON OVERLAY showing joint positions
- The person is positioned in a SIDE/PROFILE VIEW for squat assessment
- Use the skeleton markers to help assess alignment and movement quality
"""

    # Add pose detection data context if provided
    if pose_data:
        pose_context = f"""

## POSE DETECTION DATA (from QuickPose SDK - TRUST THIS DATA)
The following measurements were captured by the pose detection system during recording:

**Knee Angle Data:**
- Minimum knee angle reached: {pose_data.get('min_knee_angle', 'N/A')}°
- Maximum knee angle (standing): {pose_data.get('max_knee_angle', 'N/A')}°
- Knee angle at deepest point: {pose_data.get('knee_angle_at_deepest_point', 'N/A')}°

**Depth Assessment:**
- Depth reached below parallel (hip below knee): {"✅ YES" if pose_data.get('depth_reached_below_parallel') else "❌ NO"}
- Good depth reached (knee angle < 100°): {"✅ YES" if pose_data.get('good_depth_reached') else "❌ NO"}

**Movement Data:**
- Squats detected: {pose_data.get('squats_detected', 'N/A')}
- Person detection rate: {pose_data.get('person_detection_rate', 0) * 100:.0f}%
- Time at deepest point: {pose_data.get('time_at_deepest_point_seconds', 'N/A')}s into recording

**Real-time Feedback Shown:**
{', '.join(pose_data.get('feedback_during_recording', ['No feedback recorded']))}

⚠️ IMPORTANT: The depth and knee angle data above is MEASURED by sensors. 
Use this as GROUND TRUTH for scoring depth. Your visual analysis should CONFIRM 
or provide context, not contradict the sensor data unless you see a clear discrepancy.
"""
        base_prompt += pose_context
    else:
        base_prompt += """

## Note
No pose detection data was provided. Base your entire assessment on visual analysis only.
"""

    # Add scoring criteria
    scoring_criteria = """

## FMS Scoring Criteria

**Score 3 (Optimal):**
- Thighs reach below parallel (hip crease below knee) — USE POSE DATA FOR THIS
- Torso and tibia remain parallel (minimal forward lean < 15°) — YOU ASSESS THIS
- Heels stay flat on the ground throughout movement — YOU ASSESS THIS
- Knees track over feet with no valgus/varus collapse — YOU ASSESS THIS
- Arms maintained overhead without dropping forward — YOU ASSESS THIS (if visible)
- Smooth, controlled movement throughout — YOU ASSESS THIS

**Score 2 (Compensated):**
- Achieves depth (USE POSE DATA) but with ONE OR MORE compensations YOU observe:
  - Heels lift off the ground
  - Torso leans forward >15-20° from vertical
  - Mild knee valgus or varus
  - Arms drop forward from overhead position
  - Some loss of balance or control

**Score 1 (Dysfunctional):**
- Cannot reach parallel depth (POSE DATA shows min_knee_angle > 100°)
- OR major compensations even if depth is reached:
  - Severe forward lean (>30°)
  - Complete loss of heel contact
  - Significant knee valgus/varus
  - Unable to control the movement
  - Multiple major compensations together

**Score 0 (Pain):**
- ONLY if user explicitly reported pain (will be indicated if true)

## Scoring Decision Tree

1. First, check POSE DATA for depth:
   - If min_knee_angle < 90° → Depth achieved (potential for Score 3)
   - If min_knee_angle 90-100° → Good depth (potential for Score 2-3)
   - If min_knee_angle > 100° → Insufficient depth (likely Score 1-2)

2. Then, assess YOUR observations:
   - No compensations visible → Maintain or upgrade score
   - Minor compensations (1-2 small issues) → Score 2
   - Major compensations (multiple issues or severe) → Score 1

3. Final score = min(depth_score, compensation_score)
"""

    response_format = """

## Required Response Format (JSON)

{
  "score": <0-3>,
  "classification": "<Optimal|Compensated|Dysfunctional|Pain>",
  "pose_data_alignment": {
    "depth_confirmed": <true if your visual observation aligns with pose data depth>,
    "discrepancy_notes": "<any discrepancies between pose data and visual observation, or 'None'>"
  },
  "observations": {
    "depth": "<describe depth - REFERENCE THE POSE DATA knee angle, confirm what you see>",
    "torso": "<YOUR assessment of torso angle, estimate degrees of forward lean>",
    "heels": "<YOUR assessment - flat, lifting, or cannot determine from angle>",
    "knees": "<YOUR assessment - tracking over feet, valgus, varus, or cannot see>",
    "arms": "<YOUR assessment - overhead maintained, dropped, or not visible>"
  },
  "compensations_detected": [
    "<specific compensation you VISUALLY observed>",
    "<another compensation if any>"
  ],
  "strengths": [
    "<specific thing done well>",
    "<another strength>"
  ],
  "improvements": [
    "<actionable suggestion based on YOUR observations>",
    "<another suggestion>"
  ],
  "mobility_focus_areas": [
    "<body area needing mobility work based on compensations>",
    "<another area if applicable>"
  ],
  "summary": "<2-3 sentences: Reference the pose data depth, explain YOUR visual observations about form, justify the score>"
}

## CRITICAL RULES
1. TRUST the pose detection data for depth/knee angle - it's sensor-measured
2. YOUR job is to assess what sensors CAN'T see: torso, heels, knee valgus, arms, balance
3. If pose data says depth was reached but you see major compensations, score accordingly
4. If pose data says depth NOT reached, that alone warrants Score 1-2 even with good form
5. Be SPECIFIC about what YOU observe vs what the POSE DATA reports
6. Output ONLY valid JSON

Respond with ONLY the JSON object.
"""

    return base_prompt + scoring_criteria + response_format


# ============================================================
# AI ANALYSIS FUNCTION
# ============================================================

def analyze_squat_video(
    video_base64: str, 
    mime_type: str = "video/mp4", 
    reported_pain: bool = False,
    pose_detection_data: Optional[Dict] = None
) -> Dict:
    """Analyze squat video using Gemini Vision API with pose detection context."""
    try:
        model = genai.GenerativeModel(MODEL)
        
        # Prepare the video data for Gemini
        video_part = {
            "mime_type": mime_type,
            "data": video_base64
        }
        
        # Build the prompt with pose detection data
        prompt = build_analysis_prompt(pose_detection_data)
        
        # Add pain context if reported
        if reported_pain:
            prompt += "\n\n⚠️ USER REPORTED PAIN: The user has indicated they experienced pain during this movement. The score MUST be 0 (Pain) regardless of movement quality observed."
        
        # Log what we're sending (without the video data)
        print(f"Analyzing with pose data: {pose_detection_data is not None}")
        if pose_detection_data:
            print(f"  - Min knee angle: {pose_detection_data.get('min_knee_angle')}°")
            print(f"  - Depth reached: {pose_detection_data.get('depth_reached_below_parallel')}")
            print(f"  - Squats detected: {pose_detection_data.get('squats_detected')}")
        
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
                    2
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
            
            # Add pose data reference to result if provided
            if pose_detection_data:
                analysis_result["pose_detection_summary"] = {
                    "min_knee_angle": pose_detection_data.get("min_knee_angle"),
                    "depth_reached": pose_detection_data.get("depth_reached_below_parallel"),
                    "squats_counted": pose_detection_data.get("squats_detected")
                }
            
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
            "version": "1.2-poc",
            "features": [
                "pose_detection_integration",
                "skeleton_overlay_detection",
                "multi_angle_support",
                "pain_reporting",
                "sensor_data_fusion"
            ]
        })
    
    @app.post("/analyze")
    def analyze():
        """
        Main video analysis endpoint with pose detection data support
        
        Expected JSON body:
        {
            "video": "<base64 encoded video>",
            "mime_type": "video/mp4",
            "reported_pain": false,
            "pose_detection_data": {
                "min_knee_angle": 85.5,
                "max_knee_angle": 168.0,
                "depth_reached_below_parallel": true,
                "good_depth_reached": true,
                "knee_angle_at_deepest_point": 85.5,
                "time_at_deepest_point_seconds": 3.2,
                "squats_detected": 3,
                "person_detection_rate": 0.95,
                "feedback_during_recording": ["Good depth!", "Excellent depth! 🔥"]
            }
        }
        
        Returns:
        {
            "success": true,
            "timestamp": "...",
            "result": {
                "score": 0-3,
                "classification": "...",
                "pose_data_alignment": {...},
                "observations": {...},
                ...
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
            pose_detection_data = data.get("pose_detection_data")
            
            # Log request info
            print(f"=== Analyze Request ===")
            print(f"  mime_type: {mime_type}")
            print(f"  reported_pain: {reported_pain}")
            print(f"  video_length: {len(video_base64)} chars")
            print(f"  pose_detection_data: {'provided' if pose_detection_data else 'not provided'}")
            
            if pose_detection_data:
                print(f"  Pose data details:")
                print(f"    - min_knee_angle: {pose_detection_data.get('min_knee_angle')}")
                print(f"    - depth_reached: {pose_detection_data.get('depth_reached_below_parallel')}")
                print(f"    - squats: {pose_detection_data.get('squats_detected')}")
            
            # Perform analysis
            result = analyze_squat_video(
                video_base64, 
                mime_type, 
                reported_pain,
                pose_detection_data
            )
            
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
            pose_detection_data = data.get("pose_detection_data")
            
            print(f"Analyze-frame request - mime_type: {mime_type}, image_length: {len(image_base64)}")
            
            result = analyze_squat_video(
                image_base64, 
                mime_type, 
                reported_pain,
                pose_detection_data
            )
            
            if result["success"]:
                return jsonify({
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "result": result["analysis"],
                    "note": "Analysis based on single frame - video analysis recommended"
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
            "version": "1.2-poc",
            "endpoints": {
                "GET /health": "Health check and service info",
                "POST /analyze": "Analyze squat video with optional pose detection data",
                "POST /analyze-frame": "Analyze single image frame"
            },
            "features": {
                "pose_detection_integration": "Send pose detection metrics alongside video",
                "sensor_data_fusion": "AI uses app's knee angle data as ground truth for depth",
                "visual_analysis_focus": "AI focuses on torso, heels, knee valgus, arms, balance"
            },
            "documentation": {
                "video_format": "MP4 recommended, base64 encoded",
                "pose_data": "Include pose_detection_data object with knee angles and depth info",
                "camera_angle": "Side/profile view recommended"
            }
        })
    
    return app


# ============================================================
# VERCEL DEPLOYMENT
# ============================================================
app = create_app()

if __name__ == "__main__":
    print("Starting FMS Squat Analysis API v1.2...")
    print(f"Model: {MODEL}")
    app.run(host="0.0.0.0", port=8000, debug=True)