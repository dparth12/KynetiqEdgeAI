# """
# AI-Powered FMS Squat Analysis API - POC Version
# File: api/index.py (Vercel deployment structure)

# Updated to use pose detection metadata from the app alongside Gemini's visual analysis.
# The AI focuses on what pose detection CAN'T capture (torso lean, heel lift, knee valgus, 
# arm position, balance) while trusting the app's depth/knee angle data.
# """

# import os
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import google.generativeai as genai
# import json
# from datetime import datetime
# from typing import Dict, Optional

# # ============================================================
# # CONFIG & SETUP
# # ============================================================
# MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")

# # Configure Gemini API
# genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

# # ============================================================
# # FMS SQUAT ANALYSIS PROMPT (Dynamic based on pose data)
# # ============================================================

# def build_analysis_prompt(pose_data: Optional[Dict] = None) -> str:
#     """Build the analysis prompt, incorporating pose detection data if available."""
    
#     # Base context about what the AI should focus on
#     base_prompt = """
# You are an expert movement analyst and physical therapist specializing in Functional Movement Screening (FMS). 
# Analyze this overhead deep squat video and provide a detailed assessment.

# ## Your Role
# You are working alongside a pose detection system (QuickPose SDK) that tracks joint positions in real-time.
# The pose detection system provides ACCURATE data for:
# - Knee angle measurements
# - Whether squat depth was reached (below parallel)
# - Number of squats performed

# Your job is to analyze what the pose detection CANNOT reliably capture:
# 1. **Torso position** - Forward lean angle, uprightness
# 2. **Heel contact** - Whether heels lift off the ground
# 3. **Knee tracking** - Valgus (inward collapse) or varus (outward bow)
# 4. **Arm/overhead position** - If arms are maintained overhead
# 5. **Balance & control** - Stability throughout movement
# 6. **Overall movement quality** - Smoothness, hesitation, compensation patterns

# ## Video Context
# - The video includes a SKELETON OVERLAY showing joint positions
# - The person is positioned in a SIDE/PROFILE VIEW for squat assessment
# - Use the skeleton markers to help assess alignment and movement quality
# """

#     # Add pose detection data context if provided
#     if pose_data:
#         pose_context = f"""

# ## POSE DETECTION DATA (from QuickPose SDK - TRUST THIS DATA)
# The following measurements were captured by the pose detection system during recording:

# **Knee Angle Data:**
# - Minimum knee angle reached: {pose_data.get('min_knee_angle', 'N/A')}°
# - Maximum knee angle (standing): {pose_data.get('max_knee_angle', 'N/A')}°
# - Knee angle at deepest point: {pose_data.get('knee_angle_at_deepest_point', 'N/A')}°

# **Depth Assessment:**
# - Depth reached below parallel (hip below knee): {"✅ YES" if pose_data.get('depth_reached_below_parallel') else "❌ NO"}
# - Good depth reached (knee angle < 100°): {"✅ YES" if pose_data.get('good_depth_reached') else "❌ NO"}

# **Movement Data:**
# - Squats detected: {pose_data.get('squats_detected', 'N/A')}
# - Person detection rate: {pose_data.get('person_detection_rate', 0) * 100:.0f}%
# - Time at deepest point: {pose_data.get('time_at_deepest_point_seconds', 'N/A')}s into recording

# **Real-time Feedback Shown:**
# {', '.join(pose_data.get('feedback_during_recording', ['No feedback recorded']))}

# ⚠️ IMPORTANT: The depth and knee angle data above is MEASURED by sensors. 
# Use this as GROUND TRUTH for scoring depth. Your visual analysis should CONFIRM 
# or provide context, not contradict the sensor data unless you see a clear discrepancy.
# """
#         base_prompt += pose_context
#     else:
#         base_prompt += """

# ## Note
# No pose detection data was provided. Base your entire assessment on visual analysis only.
# """

#     # Add scoring criteria
#     scoring_criteria = """

# ## FMS Scoring Criteria

# **Score 3 (Optimal):**
# - Thighs reach below parallel (hip crease below knee) — USE POSE DATA FOR THIS
# - Torso and tibia remain parallel (minimal forward lean < 15°) — YOU ASSESS THIS
# - Heels stay flat on the ground throughout movement — YOU ASSESS THIS
# - Knees track over feet with no valgus/varus collapse — YOU ASSESS THIS
# - Arms maintained overhead without dropping forward — YOU ASSESS THIS (if visible)
# - Smooth, controlled movement throughout — YOU ASSESS THIS

# **Score 2 (Compensated):**
# - Achieves depth (USE POSE DATA) but with ONE OR MORE compensations YOU observe:
#   - Heels lift off the ground
#   - Torso leans forward >15-20° from vertical
#   - Mild knee valgus or varus
#   - Arms drop forward from overhead position
#   - Some loss of balance or control

# **Score 1 (Dysfunctional):**
# - Cannot reach parallel depth (POSE DATA shows min_knee_angle > 100°)
# - OR major compensations even if depth is reached:
#   - Severe forward lean (>30°)
#   - Complete loss of heel contact
#   - Significant knee valgus/varus
#   - Unable to control the movement
#   - Multiple major compensations together

# **Score 0 (Pain):**
# - ONLY if user explicitly reported pain (will be indicated if true)

# ## Scoring Decision Tree

# 1. First, check POSE DATA for depth:
#    - If min_knee_angle < 90° → Depth achieved (potential for Score 3)
#    - If min_knee_angle 90-100° → Good depth (potential for Score 2-3)
#    - If min_knee_angle > 100° → Insufficient depth (likely Score 1-2)

# 2. Then, assess YOUR observations:
#    - No compensations visible → Maintain or upgrade score
#    - Minor compensations (1-2 small issues) → Score 2
#    - Major compensations (multiple issues or severe) → Score 1

# 3. Final score = min(depth_score, compensation_score)
# """

#     response_format = """

# ## Required Response Format (JSON)

# {
#   "score": <0-3>,
#   "classification": "<Optimal|Compensated|Dysfunctional|Pain>",
#   "pose_data_alignment": {
#     "depth_confirmed": <true if your visual observation aligns with pose data depth>,
#     "discrepancy_notes": "<any discrepancies between pose data and visual observation, or 'None'>"
#   },
#   "observations": {
#     "depth": "<describe depth - REFERENCE THE POSE DATA knee angle, confirm what you see>",
#     "torso": "<YOUR assessment of torso angle, estimate degrees of forward lean>",
#     "heels": "<YOUR assessment - flat, lifting, or cannot determine from angle>",
#     "knees": "<YOUR assessment - tracking over feet, valgus, varus, or cannot see>",
#     "arms": "<YOUR assessment - overhead maintained, dropped, or not visible>"
#   },
#   "compensations_detected": [
#     "<specific compensation you VISUALLY observed>",
#     "<another compensation if any>"
#   ],
#   "strengths": [
#     "<specific thing done well>",
#     "<another strength>"
#   ],
#   "improvements": [
#     "<actionable suggestion based on YOUR observations>",
#     "<another suggestion>"
#   ],
#   "mobility_focus_areas": [
#     "<body area needing mobility work based on compensations>",
#     "<another area if applicable>"
#   ],
#   "summary": "<2-3 sentences: Reference the pose data depth, explain YOUR visual observations about form, justify the score>"
# }

# ## CRITICAL RULES
# 1. TRUST the pose detection data for depth/knee angle - it's sensor-measured
# 2. YOUR job is to assess what sensors CAN'T see: torso, heels, knee valgus, arms, balance
# 3. If pose data says depth was reached but you see major compensations, score accordingly
# 4. If pose data says depth NOT reached, that alone warrants Score 1-2 even with good form
# 5. Be SPECIFIC about what YOU observe vs what the POSE DATA reports
# 6. Output ONLY valid JSON

# Respond with ONLY the JSON object.
# """

#     return base_prompt + scoring_criteria + response_format


# # ============================================================
# # AI ANALYSIS FUNCTION
# # ============================================================

# def analyze_squat_video(
#     video_base64: str, 
#     mime_type: str = "video/mp4", 
#     reported_pain: bool = False,
#     pose_detection_data: Optional[Dict] = None
# ) -> Dict:
#     """Analyze squat video using Gemini Vision API with pose detection context."""
#     try:
#         model = genai.GenerativeModel(MODEL)
        
#         # Prepare the video data for Gemini
#         video_part = {
#             "mime_type": mime_type,
#             "data": video_base64
#         }
        
#         # Build the prompt with pose detection data
#         prompt = build_analysis_prompt(pose_detection_data)
        
#         # Add pain context if reported
#         if reported_pain:
#             prompt += "\n\n⚠️ USER REPORTED PAIN: The user has indicated they experienced pain during this movement. The score MUST be 0 (Pain) regardless of movement quality observed."
        
#         # Log what we're sending (without the video data)
#         print(f"Analyzing with pose data: {pose_detection_data is not None}")
#         if pose_detection_data:
#             print(f"  - Min knee angle: {pose_detection_data.get('min_knee_angle')}°")
#             print(f"  - Depth reached: {pose_detection_data.get('depth_reached_below_parallel')}")
#             print(f"  - Squats detected: {pose_detection_data.get('squats_detected')}")
        
#         # Generate response with video
#         response = model.generate_content([prompt, video_part])
#         response_text = response.text.strip()
        
#         # Clean up any markdown formatting if present
#         if '```' in response_text:
#             lines = response_text.split('\n')
#             json_lines = []
#             in_json = False
#             for line in lines:
#                 if '```json' in line:
#                     in_json = True
#                     continue
#                 elif '```' in line and in_json:
#                     in_json = False
#                     continue
#                 if in_json:
#                     json_lines.append(line)
#             if json_lines:
#                 response_text = '\n'.join(json_lines)
        
#         # Try to extract JSON if there's extra text
#         if not response_text.startswith('{'):
#             start_idx = response_text.find('{')
#             end_idx = response_text.rfind('}')
#             if start_idx != -1 and end_idx != -1:
#                 response_text = response_text[start_idx:end_idx + 1]
        
#         # Parse the JSON response
#         try:
#             analysis_result = json.loads(response_text)
            
#             # Validate and ensure all required fields exist
#             required_fields = {
#                 "score": None,
#                 "classification": "Unknown",
#                 "observations": {
#                     "depth": "Not assessed",
#                     "torso": "Not assessed",
#                     "heels": "Not assessed",
#                     "knees": "Not assessed",
#                     "arms": "Not assessed"
#                 },
#                 "compensations_detected": [],
#                 "strengths": [],
#                 "improvements": [],
#                 "mobility_focus_areas": [],
#                 "summary": "Analysis completed"
#             }
            
#             # Fill in missing fields with defaults
#             for field, default in required_fields.items():
#                 if field not in analysis_result:
#                     analysis_result[field] = default
#                 elif field == "observations" and isinstance(default, dict):
#                     for obs_field, obs_default in default.items():
#                         if obs_field not in analysis_result[field]:
#                             analysis_result[field][obs_field] = obs_default
            
#             # Validate score
#             if analysis_result.get("score") not in [0, 1, 2, 3]:
#                 # Try to infer from classification
#                 classification_map = {
#                     "Optimal": 3,
#                     "Compensated": 2,
#                     "Dysfunctional": 1,
#                     "Pain": 0
#                 }
#                 analysis_result["score"] = classification_map.get(
#                     analysis_result.get("classification"), 
#                     2
#                 )
            
#             # Ensure classification matches score
#             score_classification_map = {
#                 3: "Optimal",
#                 2: "Compensated", 
#                 1: "Dysfunctional",
#                 0: "Pain"
#             }
#             if analysis_result.get("classification") not in score_classification_map.values():
#                 analysis_result["classification"] = score_classification_map.get(
#                     analysis_result["score"], 
#                     "Compensated"
#                 )
            
#             # Add pose data reference to result if provided
#             if pose_detection_data:
#                 analysis_result["pose_detection_summary"] = {
#                     "min_knee_angle": pose_detection_data.get("min_knee_angle"),
#                     "depth_reached": pose_detection_data.get("depth_reached_below_parallel"),
#                     "squats_counted": pose_detection_data.get("squats_detected")
#                 }
            
#             return {
#                 "success": True,
#                 "analysis": analysis_result
#             }
            
#         except json.JSONDecodeError as e:
#             print(f"JSON parsing error: {e}")
#             print(f"Response text: {response_text[:500]}...")
#             return {
#                 "success": False,
#                 "error": "Failed to parse AI response as JSON",
#                 "raw_response": response_text[:1000]
#             }
            
#     except Exception as e:
#         print(f"Error in analyze_squat_video: {e}")
#         import traceback
#         traceback.print_exc()
#         return {
#             "success": False,
#             "error": str(e)
#         }


# # ============================================================
# # FLASK APP
# # ============================================================

# def create_app():
#     app = Flask(__name__)
#     CORS(app)
    
#     @app.get("/health")
#     def health():
#         return jsonify({
#             "ok": True,
#             "model": MODEL,
#             "service": "fms_squat_analysis_poc",
#             "timestamp": datetime.now().isoformat(),
#             "version": "1.2-poc",
#             "features": [
#                 "pose_detection_integration",
#                 "skeleton_overlay_detection",
#                 "multi_angle_support",
#                 "pain_reporting",
#                 "sensor_data_fusion"
#             ]
#         })
    
#     @app.post("/analyze")
#     def analyze():
#         """
#         Main video analysis endpoint with pose detection data support
        
#         Expected JSON body:
#         {
#             "video": "<base64 encoded video>",
#             "mime_type": "video/mp4",
#             "reported_pain": false,
#             "pose_detection_data": {
#                 "min_knee_angle": 85.5,
#                 "max_knee_angle": 168.0,
#                 "depth_reached_below_parallel": true,
#                 "good_depth_reached": true,
#                 "knee_angle_at_deepest_point": 85.5,
#                 "time_at_deepest_point_seconds": 3.2,
#                 "squats_detected": 3,
#                 "person_detection_rate": 0.95,
#                 "feedback_during_recording": ["Good depth!", "Excellent depth! 🔥"]
#             }
#         }
        
#         Returns:
#         {
#             "success": true,
#             "timestamp": "...",
#             "result": {
#                 "score": 0-3,
#                 "classification": "...",
#                 "pose_data_alignment": {...},
#                 "observations": {...},
#                 ...
#             }
#         }
#         """
#         try:
#             data = request.get_json(force=True) or {}
            
#             video_base64 = data.get("video")
#             if not video_base64:
#                 return jsonify({
#                     "success": False,
#                     "error": "video (base64) is required",
#                     "timestamp": datetime.now().isoformat()
#                 }), 400
            
#             mime_type = data.get("mime_type", "video/mp4")
#             reported_pain = data.get("reported_pain", False)
#             pose_detection_data = data.get("pose_detection_data")
            
#             # Log request info
#             print(f"=== Analyze Request ===")
#             print(f"  mime_type: {mime_type}")
#             print(f"  reported_pain: {reported_pain}")
#             print(f"  video_length: {len(video_base64)} chars")
#             print(f"  pose_detection_data: {'provided' if pose_detection_data else 'not provided'}")
            
#             if pose_detection_data:
#                 print(f"  Pose data details:")
#                 print(f"    - min_knee_angle: {pose_detection_data.get('min_knee_angle')}")
#                 print(f"    - depth_reached: {pose_detection_data.get('depth_reached_below_parallel')}")
#                 print(f"    - squats: {pose_detection_data.get('squats_detected')}")
            
#             # Perform analysis
#             result = analyze_squat_video(
#                 video_base64, 
#                 mime_type, 
#                 reported_pain,
#                 pose_detection_data
#             )
            
#             if result["success"]:
#                 return jsonify({
#                     "success": True,
#                     "timestamp": datetime.now().isoformat(),
#                     "result": result["analysis"]
#                 })
#             else:
#                 return jsonify({
#                     "success": False,
#                     "error": result.get("error", "Analysis failed"),
#                     "raw_response": result.get("raw_response"),
#                     "timestamp": datetime.now().isoformat()
#                 }), 500
                
#         except Exception as e:
#             print(f"Error in analyze endpoint: {e}")
#             import traceback
#             traceback.print_exc()
            
#             return jsonify({
#                 "success": False,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }), 500
    
#     @app.post("/analyze-frame")
#     def analyze_frame():
#         """
#         Analyze a single frame/image instead of video
#         """
#         try:
#             data = request.get_json(force=True) or {}
            
#             image_base64 = data.get("image")
#             if not image_base64:
#                 return jsonify({
#                     "success": False,
#                     "error": "image (base64) is required",
#                     "timestamp": datetime.now().isoformat()
#                 }), 400
            
#             mime_type = data.get("mime_type", "image/jpeg")
#             reported_pain = data.get("reported_pain", False)
#             pose_detection_data = data.get("pose_detection_data")
            
#             print(f"Analyze-frame request - mime_type: {mime_type}, image_length: {len(image_base64)}")
            
#             result = analyze_squat_video(
#                 image_base64, 
#                 mime_type, 
#                 reported_pain,
#                 pose_detection_data
#             )
            
#             if result["success"]:
#                 return jsonify({
#                     "success": True,
#                     "timestamp": datetime.now().isoformat(),
#                     "result": result["analysis"],
#                     "note": "Analysis based on single frame - video analysis recommended"
#                 })
#             else:
#                 return jsonify({
#                     "success": False,
#                     "error": result.get("error", "Analysis failed"),
#                     "timestamp": datetime.now().isoformat()
#                 }), 500
                
#         except Exception as e:
#             print(f"Error in analyze-frame endpoint: {e}")
#             import traceback
#             traceback.print_exc()
            
#             return jsonify({
#                 "success": False,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }), 500
    
#     @app.get("/")
#     def root():
#         """Root endpoint with API info"""
#         return jsonify({
#             "service": "KynetiqEdge FMS Squat Analysis API",
#             "version": "1.2-poc",
#             "endpoints": {
#                 "GET /health": "Health check and service info",
#                 "POST /analyze": "Analyze squat video with optional pose detection data",
#                 "POST /analyze-frame": "Analyze single image frame"
#             },
#             "features": {
#                 "pose_detection_integration": "Send pose detection metrics alongside video",
#                 "sensor_data_fusion": "AI uses app's knee angle data as ground truth for depth",
#                 "visual_analysis_focus": "AI focuses on torso, heels, knee valgus, arms, balance"
#             },
#             "documentation": {
#                 "video_format": "MP4 recommended, base64 encoded",
#                 "pose_data": "Include pose_detection_data object with knee angles and depth info",
#                 "camera_angle": "Side/profile view recommended"
#             }
#         })
    
#     return app


# # ============================================================
# # VERCEL DEPLOYMENT
# # ============================================================
# app = create_app()

# if __name__ == "__main__":
#     print("Starting FMS Squat Analysis API v1.2...")
#     print(f"Model: {MODEL}")
#     app.run(host="0.0.0.0", port=8000, debug=True)

# """
# AI-Powered FMS Squat Analysis API - OpenAI GPT Version
# File: api/index.py (Vercel deployment structure)

# Features intelligent phase-based frame sampling using pose detection metadata.
# Samples frames at key squat phases (start, descent, bottom, ascent, end) rather
# than uniform sampling for better analysis with fewer frames.
# """

# import os
# import base64
# import tempfile
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from openai import OpenAI
# import json
# from datetime import datetime
# from typing import Dict, Optional, List, Tuple

# # ============================================================
# # CONFIG & SETUP
# # ============================================================
# MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

# # Configure OpenAI API
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# # Frame extraction settings
# DEFAULT_FRAME_COUNT = 5
# MAX_FRAME_DIMENSION = 768  # Resize frames to this max dimension
# JPEG_QUALITY = 75


# # ============================================================
# # INTELLIGENT FRAME EXTRACTION
# # ============================================================

# def get_video_info(video_path: str) -> Tuple[float, float, int]:
#     """Get video duration, FPS, and total frame count."""
#     import cv2
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration = total_frames / fps if fps > 0 else 0
#     cap.release()
#     return duration, fps, total_frames


# def calculate_phase_timestamps(
#     duration: float,
#     pose_data: Optional[Dict] = None
# ) -> List[float]:
#     """
#     Calculate optimal timestamps to sample based on squat phases.
    
#     If pose_data is available, uses the deepest point as anchor.
#     Otherwise, estimates phases based on typical squat timing.
    
#     Returns list of timestamps (in seconds) to sample.
#     """
#     timestamps = []
    
#     if pose_data and pose_data.get('time_at_deepest_point_seconds'):
#         # We have pose data - use deepest point as anchor
#         deepest_time = pose_data['time_at_deepest_point_seconds']
        
#         # Estimate squat phases relative to deepest point
#         # Typical squat: ~1-1.5s descent, pause at bottom, ~1-1.5s ascent
        
#         # 1. Start position (beginning of video or 0.3s in)
#         start_time = min(0.3, duration * 0.05)
#         timestamps.append(start_time)
        
#         # 2. Mid-descent (halfway between start and bottom)
#         mid_descent = (start_time + deepest_time) / 2
#         timestamps.append(mid_descent)
        
#         # 3. Bottom of squat (the most important frame!)
#         timestamps.append(deepest_time)
        
#         # 4. Just after bottom (catch any compensation on the way up)
#         post_bottom = min(deepest_time + 0.3, duration - 0.2)
#         if post_bottom > deepest_time:
#             timestamps.append(post_bottom)
        
#         # 5. Mid-ascent
#         mid_ascent = (deepest_time + duration) / 2
#         if mid_ascent > post_bottom + 0.2:
#             timestamps.append(mid_ascent)
        
#         # 6. End/lockout position
#         end_time = max(duration - 0.3, duration * 0.95)
#         if end_time > timestamps[-1] + 0.2:
#             timestamps.append(end_time)
        
#         print(f"  Phase-based sampling (deepest @ {deepest_time:.2f}s):")
#         print(f"    Timestamps: {[f'{t:.2f}s' for t in timestamps]}")
        
#     else:
#         # No pose data - use estimated phases for typical squat
#         # Assume: 10% start, 30% mid-descent, 50% bottom, 70% mid-ascent, 90% end
#         phase_percentages = [0.10, 0.30, 0.50, 0.70, 0.90]
#         timestamps = [duration * p for p in phase_percentages]
#         print(f"  Uniform phase sampling (no pose data):")
#         print(f"    Timestamps: {[f'{t:.2f}s' for t in timestamps]}")
    
#     # Ensure timestamps are within bounds and sorted
#     timestamps = sorted(set(max(0, min(t, duration - 0.1)) for t in timestamps))
    
#     return timestamps


# def extract_frame_at_timestamp(
#     cap,  # cv2.VideoCapture object
#     timestamp: float,
#     fps: float
# ) -> Optional[str]:
#     """Extract a single frame at the given timestamp, return as base64 JPEG."""
#     import cv2
    
#     frame_number = int(timestamp * fps)
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#     ret, frame = cap.read()
    
#     if not ret:
#         return None
    
#     # Resize frame to reduce payload size
#     height, width = frame.shape[:2]
#     if max(height, width) > MAX_FRAME_DIMENSION:
#         scale = MAX_FRAME_DIMENSION / max(height, width)
#         new_width = int(width * scale)
#         new_height = int(height * scale)
#         frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
#     # Encode as JPEG
#     _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
#     return base64.b64encode(buffer).decode('utf-8')


# def extract_frames_intelligent(
#     video_base64: str,
#     pose_data: Optional[Dict] = None,
#     num_frames: int = DEFAULT_FRAME_COUNT
# ) -> Tuple[List[str], List[float]]:
#     """
#     Extract frames using intelligent phase-based sampling.
    
#     Returns:
#         - List of base64-encoded JPEG frames
#         - List of timestamps where frames were extracted
#     """
#     import cv2
    
#     # Decode base64 video to bytes
#     video_bytes = base64.b64decode(video_base64)
    
#     # Write to temp file
#     with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
#         tmp_file.write(video_bytes)
#         tmp_path = tmp_file.name
    
#     try:
#         # Get video info
#         duration, fps, total_frames = get_video_info(tmp_path)
#         print(f"  Video: {duration:.2f}s, {fps:.1f} FPS, {total_frames} frames")
        
#         if total_frames == 0 or fps == 0:
#             raise ValueError("Could not read video properties")
        
#         # Calculate phase-based timestamps
#         timestamps = calculate_phase_timestamps(duration, pose_data)
        
#         # If we got more timestamps than requested, prioritize key moments
#         if len(timestamps) > num_frames:
#             # Always keep: start, bottom (middle), end
#             # Then fill in others
#             if pose_data and pose_data.get('time_at_deepest_point_seconds'):
#                 deepest = pose_data['time_at_deepest_point_seconds']
#                 # Keep start, deepest, end, then closest to those
#                 priority = [timestamps[0], deepest, timestamps[-1]]
#                 remaining = [t for t in timestamps if t not in priority]
#                 # Add remaining until we hit num_frames
#                 while len(priority) < num_frames and remaining:
#                     priority.append(remaining.pop(0))
#                 timestamps = sorted(priority)
#             else:
#                 # Uniform selection from calculated timestamps
#                 step = len(timestamps) / num_frames
#                 indices = [int(i * step) for i in range(num_frames)]
#                 timestamps = [timestamps[i] for i in indices]
        
#         # Extract frames at calculated timestamps
#         cap = cv2.VideoCapture(tmp_path)
#         frames = []
#         extracted_timestamps = []
        
#         for ts in timestamps:
#             frame_b64 = extract_frame_at_timestamp(cap, ts, fps)
#             if frame_b64:
#                 frames.append(frame_b64)
#                 extracted_timestamps.append(ts)
        
#         cap.release()
        
#         print(f"  Extracted {len(frames)} frames at: {[f'{t:.2f}s' for t in extracted_timestamps]}")
        
#         return frames, extracted_timestamps
        
#     finally:
#         os.unlink(tmp_path)


# def extract_frames_uniform(
#     video_base64: str,
#     num_frames: int = DEFAULT_FRAME_COUNT
# ) -> Tuple[List[str], List[float]]:
#     """
#     Fallback: Extract frames uniformly distributed across the video.
#     Used when intelligent extraction fails or for non-squat content.
#     """
#     import cv2
    
#     video_bytes = base64.b64decode(video_base64)
    
#     with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
#         tmp_file.write(video_bytes)
#         tmp_path = tmp_file.name
    
#     try:
#         cap = cv2.VideoCapture(tmp_path)
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         duration = total_frames / fps if fps > 0 else 0
        
#         if total_frames == 0:
#             raise ValueError("Could not read video frames")
        
#         # Calculate uniform frame indices
#         frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
        
#         frames = []
#         timestamps = []
        
#         for idx in frame_indices:
#             cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
#             ret, frame = cap.read()
#             if ret:
#                 # Resize
#                 height, width = frame.shape[:2]
#                 if max(height, width) > MAX_FRAME_DIMENSION:
#                     scale = MAX_FRAME_DIMENSION / max(height, width)
#                     frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
                
#                 _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
#                 frames.append(base64.b64encode(buffer).decode('utf-8'))
#                 timestamps.append(idx / fps)
        
#         cap.release()
#         return frames, timestamps
        
#     finally:
#         os.unlink(tmp_path)


# # ============================================================
# # FMS SQUAT ANALYSIS PROMPT
# # ============================================================

# def build_analysis_prompt(
#     pose_data: Optional[Dict] = None,
#     frame_timestamps: Optional[List[float]] = None
# ) -> str:
#     """Build the analysis prompt with pose data and frame context."""
    
#     base_prompt = """
# You are an expert movement analyst and physical therapist specializing in Functional Movement Screening (FMS). 
# Analyze these overhead deep squat images (key frames extracted from video) and provide a detailed assessment.

# ## Your Role
# You are working alongside a pose detection system (QuickPose SDK) that tracks joint positions in real-time.
# The pose detection system provides ACCURATE data for:
# - Knee angle measurements
# - Whether squat depth was reached (below parallel)
# - Number of squats performed

# Your job is to analyze what the pose detection CANNOT reliably capture:
# 1. **Torso position** - Forward lean angle, uprightness
# 2. **Heel contact** - Whether heels lift off the ground
# 3. **Knee tracking** - Valgus (inward collapse) or varus (outward bow)
# 4. **Arm/overhead position** - If arms are maintained overhead
# 5. **Balance & control** - Stability throughout movement
# 6. **Overall movement quality** - Smoothness, hesitation, compensation patterns

# ## Frame Context
# """

#     # Add frame timing context
#     if frame_timestamps:
#         base_prompt += f"""
# The images are KEY FRAMES extracted at specific phases of the squat:
# - Frames provided at: {[f'{t:.2f}s' for t in frame_timestamps]}
# - These capture: start position → descent → BOTTOM (deepest point) → ascent → end
# - Pay special attention to the frame at the DEEPEST point for compensation analysis
# """
#     else:
#         base_prompt += """
# The images are frames extracted from a squat video showing the movement progression.
# """

#     base_prompt += """
# - The images may include a SKELETON OVERLAY showing joint positions
# - The person is positioned in a SIDE/PROFILE VIEW for squat assessment
# - Analyze the PROGRESSION across frames to assess movement quality
# """

#     if pose_data:
#         pose_context = f"""

# ## POSE DETECTION DATA (from QuickPose SDK - TRUST THIS DATA)
# The following measurements were captured by the pose detection system during recording:

# **Knee Angle Data:**
# - Minimum knee angle reached: {pose_data.get('min_knee_angle', 'N/A')}°
# - Maximum knee angle (standing): {pose_data.get('max_knee_angle', 'N/A')}°
# - Knee angle at deepest point: {pose_data.get('knee_angle_at_deepest_point', 'N/A')}°

# **Depth Assessment:**
# - Depth reached below parallel (hip below knee): {"✅ YES" if pose_data.get('depth_reached_below_parallel') else "❌ NO"}
# - Good depth reached (knee angle < 100°): {"✅ YES" if pose_data.get('good_depth_reached') else "❌ NO"}

# **Movement Data:**
# - Squats detected: {pose_data.get('squats_detected', 'N/A')}
# - Person detection rate: {pose_data.get('person_detection_rate', 0) * 100:.0f}%
# - Time at deepest point: {pose_data.get('time_at_deepest_point_seconds', 'N/A')}s into recording

# **Real-time Feedback Shown:**
# {', '.join(pose_data.get('feedback_during_recording', ['No feedback recorded']))}

# ⚠️ IMPORTANT: The depth and knee angle data above is MEASURED by sensors. 
# Use this as GROUND TRUTH for scoring depth. Your visual analysis should CONFIRM 
# or provide context, not contradict the sensor data unless you see a clear discrepancy.
# """
#         base_prompt += pose_context
#     else:
#         base_prompt += """

# ## Note
# No pose detection data was provided. Base your entire assessment on visual analysis only.
# """

#     scoring_criteria = """

# ## FMS Scoring Criteria

# **Score 3 (Optimal):**
# - Thighs reach below parallel (hip crease below knee) — USE POSE DATA FOR THIS
# - Torso and tibia remain parallel (minimal forward lean < 15°) — YOU ASSESS THIS
# - Heels stay flat on the ground throughout movement — YOU ASSESS THIS
# - Knees track over feet with no valgus/varus collapse — YOU ASSESS THIS
# - Arms maintained overhead without dropping forward — YOU ASSESS THIS (if visible)
# - Smooth, controlled movement throughout — YOU ASSESS THIS

# **Score 2 (Compensated):**
# - Achieves depth (USE POSE DATA) but with ONE OR MORE compensations YOU observe:
#   - Heels lift off the ground
#   - Torso leans forward >15-20° from vertical
#   - Mild knee valgus or varus
#   - Arms drop forward from overhead position
#   - Some loss of balance or control

# **Score 1 (Dysfunctional):**
# - Cannot reach parallel depth (POSE DATA shows min_knee_angle > 100°)
# - OR major compensations even if depth is reached:
#   - Severe forward lean (>30°)
#   - Complete loss of heel contact
#   - Significant knee valgus/varus
#   - Unable to control the movement
#   - Multiple major compensations together

# **Score 0 (Pain):**
# - ONLY if user explicitly reported pain (will be indicated if true)

# ## Scoring Decision Tree

# 1. First, check POSE DATA for depth:
#    - If min_knee_angle < 90° → Depth achieved (potential for Score 3)
#    - If min_knee_angle 90-100° → Good depth (potential for Score 2-3)
#    - If min_knee_angle > 100° → Insufficient depth (likely Score 1-2)

# 2. Then, assess YOUR observations:
#    - No compensations visible → Maintain or upgrade score
#    - Minor compensations (1-2 small issues) → Score 2
#    - Major compensations (multiple issues or severe) → Score 1

# 3. Final score = min(depth_score, compensation_score)
# """

#     response_format = """

# ## Required Response Format (JSON)

# {
#   "score": <0-3>,
#   "classification": "<Optimal|Compensated|Dysfunctional|Pain>",
#   "pose_data_alignment": {
#     "depth_confirmed": <true if your visual observation aligns with pose data depth>,
#     "discrepancy_notes": "<any discrepancies between pose data and visual observation, or 'None'>"
#   },
#   "observations": {
#     "depth": "<describe depth - REFERENCE THE POSE DATA knee angle, confirm what you see>",
#     "torso": "<YOUR assessment of torso angle, estimate degrees of forward lean>",
#     "heels": "<YOUR assessment - flat, lifting, or cannot determine from angle>",
#     "knees": "<YOUR assessment - tracking over feet, valgus, varus, or cannot see>",
#     "arms": "<YOUR assessment - overhead maintained, dropped, or not visible>"
#   },
#   "compensations_detected": [
#     "<specific compensation you VISUALLY observed>",
#     "<another compensation if any>"
#   ],
#   "strengths": [
#     "<specific thing done well>",
#     "<another strength>"
#   ],
#   "improvements": [
#     "<actionable suggestion based on YOUR observations>",
#     "<another suggestion>"
#   ],
#   "mobility_focus_areas": [
#     "<body area needing mobility work based on compensations>",
#     "<another area if applicable>"
#   ],
#   "summary": "<2-3 sentences: Reference the pose data depth, explain YOUR visual observations about form, justify the score>"
# }

# ## CRITICAL RULES
# 1. TRUST the pose detection data for depth/knee angle - it's sensor-measured
# 2. YOUR job is to assess what sensors CAN'T see: torso, heels, knee valgus, arms, balance
# 3. If pose data says depth was reached but you see major compensations, score accordingly
# 4. If pose data says depth NOT reached, that alone warrants Score 1-2 even with good form
# 5. Be SPECIFIC about what YOU observe vs what the POSE DATA reports
# 6. Output ONLY valid JSON

# Respond with ONLY the JSON object.
# """

#     return base_prompt + scoring_criteria + response_format


# # ============================================================
# # AI ANALYSIS FUNCTIONS
# # ============================================================

# def analyze_with_gpt(
#     images_base64: List[str],
#     reported_pain: bool = False,
#     pose_detection_data: Optional[Dict] = None,
#     frame_timestamps: Optional[List[float]] = None
# ) -> Dict:
#     """Analyze squat images using OpenAI GPT-4o Vision API."""
#     try:
#         # Build the prompt with frame context
#         prompt = build_analysis_prompt(pose_detection_data, frame_timestamps)
        
#         if reported_pain:
#             prompt += "\n\n⚠️ USER REPORTED PAIN: The user has indicated they experienced pain during this movement. The score MUST be 0 (Pain) regardless of movement quality observed."
        
#         # Build message content with multiple images
#         content = [{"type": "text", "text": prompt}]
        
#         for i, img_base64 in enumerate(images_base64):
#             # Add frame label if we have timestamps
#             if frame_timestamps and i < len(frame_timestamps):
#                 ts = frame_timestamps[i]
#                 # Identify if this is the "bottom" frame
#                 if pose_detection_data and pose_detection_data.get('time_at_deepest_point_seconds'):
#                     deepest = pose_detection_data['time_at_deepest_point_seconds']
#                     if abs(ts - deepest) < 0.5:
#                         content.append({
#                             "type": "text",
#                             "text": f"[Frame {i+1} @ {ts:.2f}s - DEEPEST POINT - analyze compensations here]"
#                         })
#                     else:
#                         content.append({
#                             "type": "text", 
#                             "text": f"[Frame {i+1} @ {ts:.2f}s]"
#                         })
            
#             content.append({
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{img_base64}",
#                     "detail": "high"
#                 }
#             })
        
#         # Log what we're sending
#         print(f"Analyzing {len(images_base64)} frames with GPT-4o")
#         if pose_detection_data:
#             print(f"  - Min knee angle: {pose_detection_data.get('min_knee_angle')}°")
#             print(f"  - Depth reached: {pose_detection_data.get('depth_reached_below_parallel')}")
#             print(f"  - Deepest point: {pose_detection_data.get('time_at_deepest_point_seconds')}s")
        
#         # Call OpenAI API
#         response = client.chat.completions.create(
#             model=MODEL,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": content
#                 }
#             ],
#             max_tokens=2000,
#             temperature=0.3
#         )
        
#         response_text = response.choices[0].message.content.strip()
        
#         # Clean up markdown formatting
#         if '```' in response_text:
#             lines = response_text.split('\n')
#             json_lines = []
#             in_json = False
#             for line in lines:
#                 if '```json' in line:
#                     in_json = True
#                     continue
#                 elif '```' in line and in_json:
#                     in_json = False
#                     continue
#                 if in_json:
#                     json_lines.append(line)
#             if json_lines:
#                 response_text = '\n'.join(json_lines)
        
#         # Extract JSON
#         if not response_text.startswith('{'):
#             start_idx = response_text.find('{')
#             end_idx = response_text.rfind('}')
#             if start_idx != -1 and end_idx != -1:
#                 response_text = response_text[start_idx:end_idx + 1]
        
#         # Parse JSON response
#         try:
#             analysis_result = json.loads(response_text)
            
#             # Validate required fields
#             required_fields = {
#                 "score": None,
#                 "classification": "Unknown",
#                 "observations": {
#                     "depth": "Not assessed",
#                     "torso": "Not assessed",
#                     "heels": "Not assessed",
#                     "knees": "Not assessed",
#                     "arms": "Not assessed"
#                 },
#                 "compensations_detected": [],
#                 "strengths": [],
#                 "improvements": [],
#                 "mobility_focus_areas": [],
#                 "summary": "Analysis completed"
#             }
            
#             for field, default in required_fields.items():
#                 if field not in analysis_result:
#                     analysis_result[field] = default
#                 elif field == "observations" and isinstance(default, dict):
#                     for obs_field, obs_default in default.items():
#                         if obs_field not in analysis_result[field]:
#                             analysis_result[field][obs_field] = obs_default
            
#             # Validate score
#             if analysis_result.get("score") not in [0, 1, 2, 3]:
#                 classification_map = {
#                     "Optimal": 3,
#                     "Compensated": 2,
#                     "Dysfunctional": 1,
#                     "Pain": 0
#                 }
#                 analysis_result["score"] = classification_map.get(
#                     analysis_result.get("classification"), 
#                     2
#                 )
            
#             # Ensure classification matches score
#             score_classification_map = {
#                 3: "Optimal",
#                 2: "Compensated", 
#                 1: "Dysfunctional",
#                 0: "Pain"
#             }
#             if analysis_result.get("classification") not in score_classification_map.values():
#                 analysis_result["classification"] = score_classification_map.get(
#                     analysis_result["score"], 
#                     "Compensated"
#                 )
            
#             # Add metadata
#             if pose_detection_data:
#                 analysis_result["pose_detection_summary"] = {
#                     "min_knee_angle": pose_detection_data.get("min_knee_angle"),
#                     "depth_reached": pose_detection_data.get("depth_reached_below_parallel"),
#                     "squats_counted": pose_detection_data.get("squats_detected")
#                 }
            
#             return {
#                 "success": True,
#                 "analysis": analysis_result
#             }
            
#         except json.JSONDecodeError as e:
#             print(f"JSON parsing error: {e}")
#             print(f"Response text: {response_text[:500]}...")
#             return {
#                 "success": False,
#                 "error": "Failed to parse AI response as JSON",
#                 "raw_response": response_text[:1000]
#             }
            
#     except Exception as e:
#         print(f"Error in analyze_with_gpt: {e}")
#         import traceback
#         traceback.print_exc()
#         return {
#             "success": False,
#             "error": str(e)
#         }


# def analyze_squat_video(
#     video_base64: str, 
#     mime_type: str = "video/mp4", 
#     reported_pain: bool = False,
#     pose_detection_data: Optional[Dict] = None,
#     num_frames: int = DEFAULT_FRAME_COUNT
# ) -> Dict:
#     """Analyze squat video with intelligent frame extraction."""
#     try:
#         print(f"=== Starting Analysis ===")
#         print(f"  Requested frames: {num_frames}")
#         print(f"  Pose data available: {pose_detection_data is not None}")
        
#         # Use intelligent extraction if we have pose data
#         if pose_detection_data:
#             frames, timestamps = extract_frames_intelligent(
#                 video_base64, 
#                 pose_detection_data, 
#                 num_frames
#             )
#         else:
#             # Fallback to uniform sampling
#             print("  No pose data - using uniform sampling")
#             frames, timestamps = extract_frames_uniform(video_base64, num_frames)
        
#         if not frames:
#             return {
#                 "success": False,
#                 "error": "Failed to extract any frames from video"
#             }
        
#         # Analyze with GPT-4o
#         return analyze_with_gpt(
#             frames, 
#             reported_pain, 
#             pose_detection_data,
#             timestamps
#         )
        
#     except Exception as e:
#         print(f"Error in analyze_squat_video: {e}")
#         import traceback
#         traceback.print_exc()
#         return {
#             "success": False,
#             "error": str(e)
#         }


# def analyze_single_image(
#     image_base64: str,
#     reported_pain: bool = False,
#     pose_detection_data: Optional[Dict] = None
# ) -> Dict:
#     """Analyze a single image with GPT-4o."""
#     return analyze_with_gpt([image_base64], reported_pain, pose_detection_data, None)


# # ============================================================
# # FLASK APP
# # ============================================================

# def create_app():
#     app = Flask(__name__)
#     CORS(app)
    
#     @app.get("/health")
#     def health():
#         return jsonify({
#             "ok": True,
#             "model": MODEL,
#             "service": "fms_squat_analysis_poc",
#             "timestamp": datetime.now().isoformat(),
#             "version": "2.1-gpt-intelligent",
#             "features": [
#                 "pose_detection_integration",
#                 "intelligent_phase_sampling",
#                 "skeleton_overlay_detection",
#                 "pain_reporting",
#                 "sensor_data_fusion"
#             ]
#         })
    
#     @app.get("/test-connection")
#     def test_connection():
#         """
#         Test OpenAI API connection with a simple completion request.
#         Returns success status and model info.
#         """
#         try:
#             # Make a minimal API call to verify connection
#             response = client.chat.completions.create(
#                 model=MODEL,
#                 messages=[
#                     {"role": "user", "content": "Say 'connected' and nothing else."}
#                 ],
#                 max_tokens=10,
#                 temperature=0
#             )
            
#             reply = response.choices[0].message.content.strip()
            
#             return jsonify({
#                 "success": True,
#                 "connected": True,
#                 "model": MODEL,
#                 "response": reply,
#                 "timestamp": datetime.now().isoformat()
#             })
            
#         except Exception as e:
#             return jsonify({
#                 "success": False,
#                 "connected": False,
#                 "model": MODEL,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }), 500
    
#     @app.post("/analyze")
#     def analyze():
#         """
#         Main video analysis endpoint with intelligent frame sampling
        
#         Expected JSON body:
#         {
#             "video": "<base64 encoded video>",
#             "mime_type": "video/mp4",
#             "reported_pain": false,
#             "num_frames": 5,
#             "pose_detection_data": {
#                 "min_knee_angle": 85.5,
#                 "max_knee_angle": 168.0,
#                 "depth_reached_below_parallel": true,
#                 "good_depth_reached": true,
#                 "knee_angle_at_deepest_point": 85.5,
#                 "time_at_deepest_point_seconds": 3.2,  // KEY: Used for intelligent sampling
#                 "squats_detected": 3,
#                 "person_detection_rate": 0.95,
#                 "feedback_during_recording": ["Good depth!", "Excellent depth! 🔥"]
#             }
#         }
        
#         The API uses time_at_deepest_point_seconds to intelligently sample frames
#         at key squat phases rather than uniform sampling.
#         """
#         try:
#             data = request.get_json(force=True) or {}
            
#             video_base64 = data.get("video")
#             if not video_base64:
#                 return jsonify({
#                     "success": False,
#                     "error": "video (base64) is required",
#                     "timestamp": datetime.now().isoformat()
#                 }), 400
            
#             mime_type = data.get("mime_type", "video/mp4")
#             reported_pain = data.get("reported_pain", False)
#             pose_detection_data = data.get("pose_detection_data")
#             num_frames = data.get("num_frames", DEFAULT_FRAME_COUNT)
            
#             print(f"=== Analyze Request ===")
#             print(f"  mime_type: {mime_type}")
#             print(f"  reported_pain: {reported_pain}")
#             print(f"  video_length: {len(video_base64)} chars")
#             print(f"  num_frames: {num_frames}")
            
#             if pose_detection_data:
#                 print(f"  Pose data:")
#                 print(f"    - min_knee_angle: {pose_detection_data.get('min_knee_angle')}")
#                 print(f"    - depth_reached: {pose_detection_data.get('depth_reached_below_parallel')}")
#                 print(f"    - deepest_point: {pose_detection_data.get('time_at_deepest_point_seconds')}s")
            
#             result = analyze_squat_video(
#                 video_base64, 
#                 mime_type, 
#                 reported_pain,
#                 pose_detection_data,
#                 num_frames
#             )
            
#             if result["success"]:
#                 return jsonify({
#                     "success": True,
#                     "timestamp": datetime.now().isoformat(),
#                     "result": result["analysis"],
#                     "sampling_method": "intelligent_phase" if pose_detection_data else "uniform"
#                 })
#             else:
#                 return jsonify({
#                     "success": False,
#                     "error": result.get("error", "Analysis failed"),
#                     "raw_response": result.get("raw_response"),
#                     "timestamp": datetime.now().isoformat()
#                 }), 500
                
#         except Exception as e:
#             print(f"Error in analyze endpoint: {e}")
#             import traceback
#             traceback.print_exc()
            
#             return jsonify({
#                 "success": False,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }), 500
    
#     @app.post("/analyze-frame")
#     def analyze_frame():
#         """Analyze a single frame/image"""
#         try:
#             data = request.get_json(force=True) or {}
            
#             image_base64 = data.get("image")
#             if not image_base64:
#                 return jsonify({
#                     "success": False,
#                     "error": "image (base64) is required",
#                     "timestamp": datetime.now().isoformat()
#                 }), 400
            
#             reported_pain = data.get("reported_pain", False)
#             pose_detection_data = data.get("pose_detection_data")
            
#             result = analyze_single_image(
#                 image_base64, 
#                 reported_pain,
#                 pose_detection_data
#             )
            
#             if result["success"]:
#                 return jsonify({
#                     "success": True,
#                     "timestamp": datetime.now().isoformat(),
#                     "result": result["analysis"],
#                     "note": "Analysis based on single frame - video analysis recommended"
#                 })
#             else:
#                 return jsonify({
#                     "success": False,
#                     "error": result.get("error", "Analysis failed"),
#                     "timestamp": datetime.now().isoformat()
#                 }), 500
                
#         except Exception as e:
#             print(f"Error in analyze-frame endpoint: {e}")
#             import traceback
#             traceback.print_exc()
            
#             return jsonify({
#                 "success": False,
#                 "error": str(e),
#                 "timestamp": datetime.now().isoformat()
#             }), 500
    
#     @app.get("/")
#     def root():
#         return jsonify({
#             "service": "KynetiqEdge FMS Squat Analysis API",
#             "version": "2.1-gpt-intelligent",
#             "model": MODEL,
#             "endpoints": {
#                 "GET /health": "Health check and service info",
#                 "GET /test-connection": "Test OpenAI API connection",
#                 "POST /analyze": "Analyze squat video with intelligent phase sampling",
#                 "POST /analyze-frame": "Analyze single image frame"
#             },
#             "intelligent_sampling": {
#                 "description": "Uses pose detection metadata for smart frame selection",
#                 "key_field": "pose_detection_data.time_at_deepest_point_seconds",
#                 "phases_captured": ["start", "mid_descent", "bottom", "post_bottom", "mid_ascent", "end"],
#                 "fallback": "Uniform sampling if pose data not provided"
#             },
#             "benefits": [
#                 "Better analysis with fewer frames",
#                 "Always captures the deepest point (where compensations are most visible)",
#                 "Lower API cost compared to uniform high-count sampling",
#                 "Phase-aware context provided to GPT for better assessment"
#             ]
#         })
    
#     return app


# # ============================================================
# # VERCEL DEPLOYMENT
# # ============================================================
# app = create_app()

# if __name__ == "__main__":
#     print("Starting FMS Squat Analysis API v2.1 (Intelligent Sampling)...")
#     print(f"Model: {MODEL}")
#     app.run(host="0.0.0.0", port=8000, debug=True)
"""
AI-Powered FMS Analysis API - Multi-Exercise Version
File: api/index.py (Vercel deployment structure)

Supports all FMS exercises:
- Lower: Functional Squat, Single Leg Balance (combined L/R), In-Line Lunge (combined L/R), Plank
- Upper: Shoulder Mobility, Push-Ups
"""

import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# ============================================================
# CONFIG & SETUP
# ============================================================
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"]) if os.environ.get("GEMINI_API_KEY") else None


# ============================================================
# ASSESSMENT CRITERIA
# KYNETIQ Edge Lower FMS v1.3, Upper FMS v1.0, Combined FMS v1.0
# (3 November 2026). The AI emits compensation tags; the rules
# below turn tags, durations and rep counts into a 0-3 score.
# ============================================================

RULE_VERSION = "lower-1.3+upper-1.0+combined-1.0"

# Engineering starting points from the specs. Keep configurable;
# calibrate against clinician-rated videos before production.
PERSISTENT_SECONDS = float(os.environ.get("PERSISTENT_SECONDS", "0.5"))

SHARED_RULES = f"""
## How to observe
- You are a movement observer supporting coach review. Describe visible movement behaviour only.
  Never name an anatomical cause, a weakness, a pathology, an injury risk or a clearance.
- Use ONLY the compensation tags listed for this exercise. Each tag needs a severity
  (transient, mild, moderate, major), a confidence from 0.00 to 1.00, the time it was first
  seen in seconds from the start of the active task, and its total duration in seconds.
- Severity: transient = brief, does not alter task completion and does not repeat.
  mild = small, recoverable, main task criteria kept. moderate = clear and repeated but
  controlled, task still completed. major = causes task failure, substantial alignment loss,
  a step, touchdown, hand support, foot slide, lost support point, or inability to recover.
- A deviation is persistent when it lasts at least {PERSISTENT_SECONDS} seconds or appears in
  at least two of three repetitions. Transient deviations never lower the score unless they
  cause support loss or balance failure. Persistent mild may contribute to a 2. Moderate caps
  the exercise at 2. Major caps the exercise at 1.
- A step, touchdown, hand support, foot slide or lost support point is an immediate major
  event even when brief.
- A one-frame landmark jump with stable neighbouring frames is a tracking artifact. Ignore it.
- Do not double-penalise linked observations that arise from one event (for example hip sag
  and lumbar extension together).
- Never infer pain from a face, a grimace or a movement. Pain comes only from the athlete's
  own report, which is supplied to you separately.
- If a required landmark is occluded or outside the frame, or the required view is missing,
  so that a primary criterion cannot be evaluated, set rateable to false with a short reason
  and leave score null. Never score an invalid recording as a movement failure.
- Forward knee travel beyond the toes is not a fault. Small ankle corrections during a
  balance hold are normal and are not faults.
- Markerless 2D video supports repeatable observation; do not present it as exact 3D
  kinematics. The on-device joint angles supplied below are sensor data to support, not
  replace, what you see.
"""

# Tags whose spec implication is review-only or a secondary quality flag.
# They are reported to the coach but never cap the score by themselves.
NON_SCORING_TAGS = {
    "SM_SCAPULAR_BORDER_PROMINENCE", "SM_SCAPULAR_MOTION_ASYMMETRY",
    "SM_UPPER_ELBOW_FLARE", "SM_LOWER_ELBOW_FLARE",
    "PU_CERVICAL_EXTENSION", "PU_CERVICAL_FLEXION", "PU_SHOULDER_ELEVATION",
    "PU_TEMPO_BREAKDOWN",
    "PLANK_CERVICAL_EXTENSION", "PLANK_CERVICAL_FLEXION", "PLANK_SUPPORT_ASYMMETRY",
}

# "Secondary; persistent finding may cap at 2": never caps below 2.
SECONDARY_TAGS = {"PU_FORWARD_HEAD", "PLANK_FORWARD_HEAD"}

# Immediate major events regardless of the severity the model assigns.
ALWAYS_MAJOR_TAGS = {
    "SQUAT_BALANCE_LOSS", "SQUAT_STANCE_CHANGE",
    "SLB_FOOT_REPOSITION", "SLB_TOUCHDOWN",
    "LUNGE_BALANCE_LOSS", "LUNGE_STEP_OFF_LINE",
    "PLANK_CONTACT_LOSS",
    "SM_FOOT_REPOSITIONING",
}

# Push-up repetition validity: these invalidate a rep but do not cap by themselves.
INVALID_REP_TAGS = {
    "PU_TRUNK_LAG", "PU_HIP_LEAD", "PU_BODY_NOT_UNIT", "PU_INCOMPLETE_DEPTH",
    "PU_INCOMPLETE_LOCKOUT", "PU_HAND_REPOSITIONING", "PU_FOOT_REPOSITIONING",
    "PU_KNEE_CONTACT", "PU_CHEST_REST", "PU_REBOUND", "PU_PAUSE_FAILURE",
}

EXERCISE_CRITERIA = {
    "functional_squat": {
        "name": "Functional Squat",
        "category": "lower_fms",
        "mode": "reps",
        "views": "Frontal and sagittal recordings of the same three repetitions.",
        "focus": """
Analyse three repetitions. Score each repetition 0-3 and report them in rep_scores; the
exercise score is the median. Pain overrides all repetitions.

Sagittal view (use for): depth, trunk-to-tibia relationship, heel rise, lumbopelvic position,
balance. Frontal view (use for): knee-to-foot tracking, pelvic shift, trunk lean, asymmetry,
foot repositioning. A frontal view must not decide depth or lumbar flexion by itself; a
sagittal view must not decide frontal-plane knee displacement by itself.

Trunk lean is judged relative to the tibia, never as an absolute angle from vertical: use a
trunk inclination exceeding tibial inclination by more than 10-15 degrees as the initial mild
range. The on-device knee angles tell you depth: a minimum knee angle below about 90 degrees
means the hip crease passed below the knee.
""",
        "tags": [
            ("SQUAT_DEPTH_LIMITED", "Hip crease does not reach approximately knee height or the femur remains above horizontal.", "Score 1 when the athlete cannot reach parallel across the scored repetitions"),
            ("SQUAT_EXCESSIVE_TRUNK_LEAN", "Trunk becomes substantially more horizontal than the tibia.", "Score 2 when mild or moderate; score 1 when major or combined with task failure"),
            ("SQUAT_TRUNK_TIBIA_MISMATCH", "Trunk forward inclination exceeds tibial inclination by more than the allowance.", "10 to 15 degrees is the initial mild range"),
            ("SQUAT_HEEL_RISE", "One or both heels lose floor contact during descent or at the bottom.", "Score 2 when intermittent; score 1 when major or persistent"),
            ("SQUAT_LUMBOPELVIC_FLEXION", "Gross posterior pelvic rotation or visible lumbar rounding near the bottom.", "Score 2 when mild; score 1 when major"),
            ("SQUAT_LUMBOPELVIC_EXTENSION", "Excessive lumbar arching or rib flare during the repetition.", "Score 2 when persistent; score 1 when major"),
            ("SQUAT_BALANCE_LOSS", "The athlete steps, repositions a foot, uses support, or cannot complete the repetition.", "Immediate major event and maximum score of 1"),
            ("SQUAT_UNCONTROLLED_TEMPO", "Rapid drop, rebound, or inability to control the bottom or ascent.", "Score 2 when the task remains complete; score 1 when control is lost"),
            ("SQUAT_KNEE_MEDIAL", "Knee centre moves clearly medial to the corresponding foot line.", "Score 2 when mild or moderate; score 1 when major and persistent"),
            ("SQUAT_KNEE_LATERAL", "Knee centre moves excessively lateral to the corresponding foot line.", "Score 2 when mild or moderate; score 1 when major and persistent"),
            ("SQUAT_ASYMMETRIC_KNEE_TRACKING", "One knee deviates more than the other across the same phase.", "Set asymmetry; lower score according to deviation severity"),
            ("SQUAT_LATERAL_PELVIC_SHIFT", "Pelvic midpoint shifts noticeably toward one side.", "Score 2 when controlled; score 1 when major or with balance loss"),
            ("SQUAT_PELVIC_OBLIQUITY", "One hip remains visibly higher than the other.", "Score 2 when persistent; score 1 when major"),
            ("SQUAT_LATERAL_TRUNK_LEAN", "Shoulder midpoint shifts relative to the pelvic midpoint.", "Score 2 when persistent; score 1 when major"),
            ("SQUAT_FOOT_ROTATION_CHANGE", "A foot turns substantially inward or outward from its starting orientation.", "Score 2 when repeated; score 1 when it becomes a step or major reposition"),
            ("SQUAT_FOOT_CONTACT_LOSS", "The medial or lateral foot border visibly loses contact.", "Score 2 when intermittent; score 1 when major or persistent"),
            ("SQUAT_STANCE_CHANGE", "A foot slides, pivots, steps, or changes stance width.", "Immediate major event when the base of support is lost"),
        ],
        "scoring": """
3: Hip crease reaches below knee level; heels remain grounded; trunk approximately parallel
   with the tibia or more vertical; knees track with the feet; pelvis stays centred; feet stay
   fixed; movement controlled; no meaningful lumbopelvic deviation or balance loss.
2: Completes the squat to at least approximately parallel, but with one or more persistent
   mild or moderate compensations (trunk-to-tibia mismatch, intermittent heel rise, mild knee
   displacement, lateral shift, lumbopelvic deviation, foot-contact loss, uneven control).
1: Cannot reach parallel, has a major or persistent compensation, steps or repositions the
   feet, loses balance, descends without control, or shows multiple moderate compensations.
0-NP (zero_reason "unable"): cannot safely assume the start position or initiate the squat.
""",
        "observations_keys": ["depth", "torso", "heels", "knees", "overall"],
    },

    "single_leg_balance": {
        "name": "Single-Leg Balance",
        "category": "lower_fms",
        "mode": "hold",
        "hold_seconds": 20,
        "views": "Frontal and sagittal recordings of the same hold.",
        "focus": """
This recording is ONE leg (the stance leg is named in the exercise title). Start the clean-hold
clock when the non-stance foot leaves the floor. Stop it at the first touchdown, contact with the
stance leg for support, stance-foot hop or slide, external support, or an unrecovered major
deviation. Report that time as clean_duration_seconds (cap at the task length).

Sagittal view (use for): anterior-posterior sway, trunk lean, hip and knee strategy, heel or toe
rise. Frontal view (use for): pelvic drop or hike, mediolateral sway, trunk lean, knee alignment,
touchdown. Normal small ankle corrections are expected and are not faults.
""",
        "tags": [
            ("SLB_ANTERIOR_POSTERIOR_SWAY", "Repeated or large forward and backward displacement of the shoulders or pelvis over the stance foot.", "Score 2 when persistent but controlled; score 1 when unrecovered"),
            ("SLB_FORWARD_TRUNK_LEAN", "Sustained forward trunk displacement rather than a generally upright position.", "Score 2 when moderate; score 1 when major"),
            ("SLB_HIP_FLEXION_STRATEGY", "Trunk and pelvis remain flexed to maintain the hold.", "Score 2 when persistent; score 1 when it prevents control"),
            ("SLB_KNEE_FLEXION_COLLAPSE", "The stance knee progressively bends or buckles without recovery.", "Major event when substantial or unrecovered"),
            ("SLB_HEEL_RISE", "The stance heel repeatedly or continuously lifts.", "Score 2 when controlled; score 1 when associated with balance loss"),
            ("SLB_TOE_RISE", "The forefoot lifts and weight shifts toward the heel.", "Score 2 when repeated; score 1 when associated with balance loss"),
            ("SLB_FOOT_REPOSITION", "The stance foot slides, pivots, hops, or steps.", "Immediate major event and end of clean-hold time"),
            ("SLB_TOUCHDOWN", "The non-stance foot touches the floor or stance limb.", "Immediate major event and end of clean-hold time"),
            ("SLB_CONTRALATERAL_PELVIC_DROP", "The non-stance hip drops below the stance hip.", "5 to 10 degrees is the initial mild range; more than 10 degrees and persistent is major"),
            ("SLB_PELVIC_HIKE", "The non-stance hip rises excessively.", "Score according to persistence and magnitude"),
            ("SLB_LATERAL_TRUNK_LEAN", "The trunk leans toward or away from the stance limb.", "Score 2 when controlled; score 1 when major"),
            ("SLB_MEDIOLATERAL_SWAY", "The pelvis or shoulder midpoint repeatedly moves side to side.", "Score 2 when persistent but controlled; score 1 when unrecovered"),
            ("SLB_KNEE_MEDIAL", "The stance knee moves medial to the foot line.", "Score 2 when mild or moderate; score 1 when major"),
            ("SLB_KNEE_LATERAL", "The stance knee moves excessively lateral to the foot line.", "Score 2 when mild or moderate; score 1 when major"),
            ("SLB_FOOT_EDGE_LOADING", "Repeated loading on the medial or lateral foot border.", "Do not lower for a normal ankle strategy; score only when excessive or with control loss"),
            ("SLB_ARM_STRATEGY", "Large repeated arm movements are used to prevent balance loss.", "Score 2 when repeated; score 1 when unable to recover"),
        ],
        "scoring": """
3: Clean hold of at least 20 seconds with only minimal controlled sway, generally upright trunk,
   pelvic drop or hike below approximately 10 degrees, stable knee-to-foot alignment, full-foot
   contact, no touchdown or repositioning.
2: Clean hold of 10 to 19 seconds, or 20 seconds with persistent mild or moderate sway, 5 to 10
   degrees of pelvic drop or hike, mild trunk lean, mild knee deviation, intermittent heel or
   toe rise, or moderate arm recovery.
1: Clean hold under 10 seconds, or a touchdown, hop, slide, persistent pelvic deviation over
   approximately 10 degrees, major trunk lean, major knee displacement, frequent uncontrolled
   wobbling, or an unrecovered balance reaction.
0-NP (zero_reason "unable"): cannot safely assume single-leg stance or lift the non-stance foot.
""",
        "observations_keys": ["stability", "hipDrop", "trunkPosition", "duration", "overall"],
    },

    "inline_lunge": {
        "name": "In-Line Lunge",
        "category": "lower_fms",
        "mode": "reps",
        "views": "Frontal and sagittal recordings of the same three repetitions.",
        "focus": """
This recording is ONE lead leg (named in the exercise title). Analyse three repetitions, score
each 0-3 and report them in rep_scores; the side score is the median.

Sagittal view (use for): depth target (back knee to the floor immediately behind the front
heel), trunk lean, front-heel contact, lumbopelvic position, control. Frontal view (use for):
knee-to-foot tracking, pelvic and trunk deviation, step off the line, frontal sway. An initial
stride that does not match the prescribed setup invalidates that repetition rather than
lowering the score: tag LUNGE_INCORRECT_STRIDE and exclude it from rep_scores.
""",
        "tags": [
            ("LUNGE_DEPTH_LIMITED", "The back knee does not descend to the target immediately behind the front heel.", "Score 2 when inconsistent; score 1 when the target is repeatedly not achieved"),
            ("LUNGE_FORWARD_TRUNK_LEAN", "The trunk inclines substantially forward instead of remaining relatively upright.", "Score 2 when mild or moderate; score 1 when major"),
            ("LUNGE_LUMBOPELVIC_FLEXION", "Gross lumbar rounding or posterior pelvic rotation occurs.", "Score according to persistence and magnitude"),
            ("LUNGE_LUMBOPELVIC_EXTENSION", "Excessive lumbar arching or rib flare occurs.", "Score according to persistence and magnitude"),
            ("LUNGE_FRONT_HEEL_RISE", "The front heel loses contact during descent or ascent.", "Score 2 when intermittent; score 1 when major or persistent"),
            ("LUNGE_BACK_FOOT_CONTACT_LOSS", "The back foot shifts, rolls, or loses required contact.", "Score 2 when controlled; score 1 when support is lost"),
            ("LUNGE_INCORRECT_STRIDE", "The initial spacing does not match the prescribed setup.", "Invalidate the repetition and request setup correction"),
            ("LUNGE_UNCONTROLLED_DESCENT", "The athlete drops rapidly, impacts the floor, or cannot reverse under control.", "Score 2 when task remains complete; score 1 when control is lost"),
            ("LUNGE_BALANCE_LOSS", "The athlete steps, hops, uses support, or cannot return to the start.", "Immediate major event and maximum score of 1"),
            ("LUNGE_FRONT_KNEE_MEDIAL", "The front knee moves medial to the front-foot line.", "Score 2 when mild or moderate; score 1 when major"),
            ("LUNGE_FRONT_KNEE_LATERAL", "The front knee moves excessively lateral to the foot line.", "Score 2 when mild or moderate; score 1 when major"),
            ("LUNGE_REAR_KNEE_DEVIATION", "The rear knee moves clearly inside or outside the intended line.", "Score according to persistence and magnitude"),
            ("LUNGE_LATERAL_TRUNK_LEAN", "The shoulder midpoint shifts laterally relative to the pelvis.", "Score 2 when controlled; score 1 when major"),
            ("LUNGE_PELVIC_DROP_HIKE", "The pelvis becomes visibly unlevel.", "Score 2 when mild or moderate; score 1 when major"),
            ("LUNGE_GROSS_TRUNK_ROTATION", "The shoulders and pelvis visibly rotate away from the intended line.", "Score 2 when clear and repeated; score 1 when major"),
            ("LUNGE_STEP_OFF_LINE", "Either foot moves off the designated line.", "Immediate major event when the base of support changes"),
            ("LUNGE_FOOT_ROTATION", "Either foot turns substantially inward or outward.", "Score 2 when repeated; score 1 when it becomes a major reposition"),
            ("LUNGE_FRONTAL_SWAY", "The trunk or pelvis repeatedly oscillates side to side.", "Score 2 when controlled; score 1 when unrecovered"),
        ],
        "scoring": """
3: At least two of three controlled repetitions; trunk upright; front heel grounded; back knee
   reaches the target behind the front heel; knees track with the feet; pelvis controlled; feet
   stay on the line; no gross rotation, stepping, or balance loss.
2: Completes the movement with persistent mild or moderate trunk lean, intermittent heel rise,
   mild knee deviation, pelvic drop or rotation, frontal sway, minor foot rotation, slight line
   deviation, inconsistent depth target, or reduced but adequate control.
1: Cannot complete a controlled repetition, repeatedly misses the depth target, major heel rise
   or knee deviation, steps off the line, repositions a foot, major trunk or pelvic deviation,
   repeated balance loss, or cannot return under control.
0-NP (zero_reason "unable"): cannot safely assume the in-line start position or initiate the lunge.
""",
        "observations_keys": ["alignment", "kneeTracking", "balance", "trunkPosition", "overall"],
    },

    "plank": {
        "name": "Forward Plank",
        "category": "shared",
        "mode": "hold",
        "hold_seconds": 30,
        "views": "Sagittal (side-facing) recording is required. A frontal recording is optional and only adds symmetry tags.",
        "focus": """
Score the plank from the side-facing sagittal recording only. If the only recording is
front-facing, or the shoulders, elbows, hips, knees, ankles and feet are not all visible, set
rateable to false ("sagittal side view required") and leave score null.

Run a clean-hold clock from the start of the hold to the first support-point loss, knee
flexion used to continue, base repositioning, or unrecovered major position loss; report it as
clean_duration_seconds (cap at 30). Judge hip sag or pike against the shoulder-to-ankle line.
This one plank score is shared by the Lower and Upper assessments.
""",
        "tags": [
            ("PLANK_HIP_SAG", "Hip midpoint falls below the shoulder-to-ankle line.", "Score 2 when mild or intermittent; score 1 when major or persistent"),
            ("PLANK_HIP_PIKE", "Hip midpoint rises substantially above the shoulder-to-ankle line.", "Score 2 when mild or intermittent; score 1 when major or persistent"),
            ("PLANK_LUMBAR_EXTENSION", "Visible lumbar arching with anterior pelvic tilt.", "Score according to persistence and magnitude"),
            ("PLANK_LUMBAR_FLEXION", "Lumbar curve moves into marked flexion or posterior pelvic tilt.", "Score according to persistence and magnitude"),
            ("PLANK_TRUNK_FLEXION", "Upper-back rounding or loss of the straight trunk position.", "Score according to persistence and magnitude"),
            ("PLANK_THORACIC_FLEXION", "Thoracic kyphosis increases from the starting position.", "Moderate caps at 2; major at 1"),
            ("PLANK_THORACIC_COLLAPSE", "Thorax sinks relative to the shoulder girdle.", "Major finding may cap at 1"),
            ("PLANK_FORWARD_BACKWARD_SHIFT", "Shoulders move substantially forward or backward relative to the elbows.", "Score 2 when repeated; score 1 when support or posture is lost"),
            ("PLANK_REPEATED_SAGITTAL_SWAY", "Trunk or pelvis repeatedly oscillates forward, backward or vertically.", "Repeated caps at 2; major at 1"),
            ("PLANK_ELBOW_POSITION", "Elbows are not approximately beneath the shoulders.", "Setup error; score only if it persists after setup correction"),
            ("PLANK_FORWARD_HEAD", "Head moves anterior to the trunk line.", "Secondary; persistent finding may cap at 2"),
            ("PLANK_CERVICAL_EXTENSION", "Chin lifts beyond the starting alignment.", "Secondary quality flag"),
            ("PLANK_CERVICAL_FLEXION", "Head drops below the trunk reference line.", "Secondary quality flag"),
            ("PLANK_KNEE_FLEXION", "One or both knees bend to continue the hold.", "Score 1 when used persistently to maintain the test"),
            ("PLANK_BASE_REPOSITIONING", "Elbow, forearm or foot changes position.", "Major event; cap at 1 when used to continue"),
            ("PLANK_CONTACT_LOSS", "A forearm, toe, or other required support point loses contact.", "Immediate major event and end of clean-hold time"),
            ("PLANK_EARLY_TERMINATION", "The athlete stops before the 30-second target.", "Score according to clean-hold duration"),
            ("PLANK_SHOULDER_ASYMMETRY", "(frontal only) One shoulder remains visibly higher than the other.", "Coach observation; lower score only when persistent and meaningful"),
            ("PLANK_PELVIC_OBLIQUITY", "(frontal only) One hip remains higher than the other.", "Coach observation; lower score according to severity"),
            ("PLANK_LATERAL_SHIFT", "(frontal only) Trunk or pelvic midpoint shifts laterally.", "Coach observation; lower score according to severity"),
            ("PLANK_SUPPORT_ASYMMETRY", "(frontal only) Elbows, forearms, knees, or feet show unequal support placement.", "Setup correction or coach observation"),
        ],
        "scoring": """
3: Maintains the position for 30 seconds; shoulders, hips, knees and ankles remain approximately
   aligned; no meaningful hip sag or pike; spine and pelvis generally neutral; elbows beneath
   the shoulders; knees extended; all support points grounded.
2: Clean position for 20 to 29 seconds, or 30 seconds with mild or intermittent hip sag, piking,
   lumbar extension, trunk flexion, forward-backward shift, elbow-position deviation, or
   correctable asymmetry.
1: Clean hold under 20 seconds, or persistent major hip sag or piking, major spinal deviation,
   knee flexion used to continue, support-point loss, repeated position loss, or need for
   assistance.
0-NP (zero_reason "unable"): cannot safely assume the plank start position or lift into the test.
""",
        "observations_keys": ["spineAlignment", "hipPosition", "holdTime", "stability", "overall"],
    },

    "shoulder_mobility": {
        "name": "Shoulder Mobility Functional Combined Reach",
        "category": "upper_fms",
        "mode": "shoulder",
        "views": "Posterior frontal view: the athlete stands with the back to the camera.",
        "focus": """
This recording is ONE arm configuration (the arm reaching over the top is named in the exercise
title; the other arm reaches up behind the lower back). The test combines shoulder rotation with
abduction, adduction, extension, scapular movement, elbow motion, wrist position and trunk
control. Do not label it as isolated internal or external rotation.

Hand-length reference: the distance from the distal wrist crease to the tip of the third digit.
Estimate the minimum fist-to-fist distance divided by that hand length and report it as
normalized_hand_gap. When ON-DEVICE POSE DATA gives a fist gap in hand lengths, use that value as
normalized_hand_gap instead of estimating. Both hands must stay as closed fists; an open hand, extended fingers or a
hand flattened against the back invalidates the trial (tag SM_HAND_POSITION_CHANGE and set
rateable false with reason "hand opened; repeat trial"). Required visibility: both fists,
wrists, shoulders, elbows, scapular borders, spinal midline and feet.
""",
        "tags": [
            ("SM_EXCESS_HAND_GAP", "Minimum fist-to-fist distance exceeds the applicable hand-length band.", "Score by normalized gap"),
            ("SM_HAND_WALKING", "Fingers or hands make repeated movements after the initial reach.", "Repeat once; persistent finding caps at 1"),
            ("SM_MULTIPLE_REACH_ATTEMPTS", "More than one reach is used to obtain the endpoint.", "Cap at 2; cap at 1 when repeated and substantial"),
            ("SM_TRUNK_ROTATION", "Shoulder line or spinal midline rotates toward one side.", "Moderate caps at 2; major caps at 1"),
            ("SM_LATERAL_TRUNK_FLEXION", "Trunk midline bends toward either hand.", "Moderate caps at 2; major caps at 1"),
            ("SM_SHOULDER_ELEVATION", "One or both shoulders move toward the ears during the reach.", "Persistent finding caps at 2"),
            ("SM_UNEVEN_SHOULDER_HEIGHT", "Persistent left-right difference in acromion height.", "Coach flag; lower score by severity"),
            ("SM_SCAPULAR_BORDER_PROMINENCE", "Medial border or inferior angle becomes visibly prominent.", "Review-only; no automatic deduction"),
            ("SM_SCAPULAR_MOTION_ASYMMETRY", "One scapula moves earlier or substantially differently.", "Review-only; no automatic deduction"),
            ("SM_UPPER_ELBOW_FLARE", "Upper elbow moves substantially laterally.", "Secondary quality flag"),
            ("SM_LOWER_ELBOW_FLARE", "Lower elbow separates substantially from the torso.", "Secondary quality flag"),
            ("SM_WRIST_COMPENSATION", "Marked wrist position reduces the apparent fist distance.", "Repeat or cap at 2; major caps at 1"),
            ("SM_HAND_POSITION_CHANGE", "Fist opens, fingers extend, or hand flattens against the back.", "Measurement invalid for that trial; repeat"),
            ("SM_INCOMPLETE_HOLD", "Final position cannot be maintained for the capture interval.", "Cap at 1 when persistent"),
            ("SM_FOOT_REPOSITIONING", "Athlete steps or changes foot position.", "Immediate major event; cap at 1"),
        ],
        "scoring": """
3: Trial completed; normalized hand gap no more than 1.0 hand length; one smooth reach;
   standardised fists; no hand walking or persistent meaningful compensation.
2: Gap greater than 1.0 but no more than 1.5 hand lengths; or a 3-level distance achieved with a
   persistent mild or moderate compensation.
1: Gap exceeds 1.5 hand lengths; or repeated attempts, persistent hand walking, major trunk
   substitution, altered hand position, or inability to hold the endpoint.
0-NP (zero_reason "unable"): cannot safely position either arm behind the back.
""",
        "observations_keys": ["fistDistance", "shoulderRange", "overall"],
    },

    "pushups": {
        "name": "Push-Up",
        "category": "upper_fms",
        "mode": "pushup",
        "views": "Sagittal side view of one continuous set.",
        "focus": """
Count VALID repetitions only and report valid_rep_count, plus the total attempted as
attempted_rep_count. The variation (standard or knee) and the scoring profile (men or women)
were selected by the athlete before recording and are supplied below. Never infer them from the
video; if the video clearly shows a different variation than the one selected, say so in
coach_observation but still count against the selected variation.

A valid standard repetition: hands and feet stay in their original support positions and knees
stay off the floor; shoulder, trunk, pelvis, knees and ankles move as one unit without major
sag, pike or spinal lag; the athlete reaches the bottom position and returns to full lockout at
the top; no chest rest, rebound, prolonged pause or repositioning.

A valid knee repetition: knees stay on the surface and shoulder, trunk, pelvis and knee form the
alignment line; no hinging backward at the hips and the feet do not become an active support
point; trunk and pelvis rise together through the bottom and top positions; hands and knees do
not slide; no chest rest, rebound or prolonged pause.

Three consecutive invalid repetitions terminate the set; stop counting there. Report every rep
that was invalid with its rep_index on the tag that invalidated it. A sagittal view cannot
reliably judge elbow flare, hand width, trunk rotation, lateral shift or scapular winging; do
not report those.
""",
        "tags": [
            ("PU_LUMBAR_EXTENSION", "Pelvis drops below the shoulder-to-ankle or shoulder-to-knee line with increased lumbar extension.", "Score-critical; moderate caps at 2, major at 1"),
            ("PU_HIP_PIKE", "Pelvis rises above the applicable body line.", "Score-critical; moderate caps at 2, major at 1"),
            ("PU_TRUNK_LAG", "Upper trunk initiates ascent while the pelvis remains low.", "Invalid repetition; persistent finding caps at 1"),
            ("PU_HIP_LEAD", "Pelvis rises before the shoulders.", "Invalid repetition; persistent finding caps at 1"),
            ("PU_BODY_NOT_UNIT", "Shoulder, trunk, pelvis and lower body do not move together.", "Invalid repetition"),
            ("PU_INCOMPLETE_DEPTH", "Athlete reverses before the protocol-defined bottom position.", "Do not count repetition"),
            ("PU_INCOMPLETE_LOCKOUT", "Elbow does not return to the protocol-defined top position.", "Do not count repetition"),
            ("PU_FORWARD_HEAD", "Head translates anteriorly relative to the trunk.", "Secondary; persistent finding may cap at 2"),
            ("PU_CERVICAL_EXTENSION", "Chin lifts and cervical extension increases.", "Secondary quality flag"),
            ("PU_CERVICAL_FLEXION", "Head drops below the trunk reference line.", "Secondary quality flag"),
            ("PU_THORACIC_FLEXION", "Upper-back rounding increases substantially.", "Moderate caps at 2; major at 1"),
            ("PU_SHOULDER_ELEVATION", "Shoulder moves toward the ear under load.", "Secondary quality flag"),
            ("PU_EXCESSIVE_FORWARD_SHIFT", "Shoulder translates substantially beyond the hand during descent.", "Lower score by severity"),
            ("PU_HAND_REPOSITIONING", "Hand slides, rotates or lifts after the set begins.", "Invalid repetition; repeated event caps at 1"),
            ("PU_FOOT_REPOSITIONING", "Foot slides or changes support position.", "Invalid repetition; repeated event caps at 1"),
            ("PU_KNEE_CONTACT", "Knee contacts the floor during standard mode.", "Invalid standard repetition; offer knee pathway"),
            ("PU_CHEST_REST", "Chest or abdomen unloads on the floor.", "Invalid repetition"),
            ("PU_REBOUND", "Visible floor contact or momentum assists the reversal.", "Invalid repetition"),
            ("PU_PAUSE_FAILURE", "Athlete stops mid-repetition or cannot return to the top.", "Invalid repetition; may end set"),
            ("PU_TEMPO_BREAKDOWN", "Repetition duration increases substantially relative to earlier valid repetitions.", "Fatigue flag; score by valid count and form"),
        ],
        "scoring": """
The base score is computed from valid_rep_count by the server using the selected variation and
profile (standard: men 15+ = 3, 5-14 = 2, fewer = knee retest; women 10+ = 3, 3-9 = 2; knee:
men 15+ = 2, 1-14 = 1; women 10+ = 2, 1-9 = 1; no valid rep = 0-NP). Quality modifiers then
apply: persistent mild or moderate compensation caps at 2; major sag, pike, body-segment lag,
repeated repositioning or repeated invalid repetitions cap at 1. Your job is the valid count,
the tags and the observation; still fill in score with your best estimate.
""",
        "observations_keys": ["repCount", "bodyPosition", "depth", "control", "overall"],
    },
}


# ============================================================
# PROMPT BUILDING
# ============================================================

def _fmt(val, decimals=0):
    if val is None or val == 'N/A':
        return 'N/A'
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return 'N/A'


def format_pose_data(view_types: List[str], pose_data_list: List[Dict]) -> str:
    """On-device joint angles captured during recording, one block per view."""
    if not pose_data_list:
        return ""
    text = "\n## ON-DEVICE POSE DATA (joint angles measured during recording)\n"
    text += "Use these as sensor support for depth, hip level, hold time and range.\n"
    for i, pose_data in enumerate(pose_data_list):
        view_name = view_types[i] if i < len(view_types) else f"View {i+1}"
        text += f"\n**{view_name.replace('_', ' ').title()}:**\n"
        left_knee = pose_data.get('left_knee', {})
        right_knee = pose_data.get('right_knee', {})
        if left_knee or right_knee:
            text += (f"- Left Knee: {_fmt(left_knee.get('min'))}° - {_fmt(left_knee.get('max'))}° "
                     f"(deepest: {_fmt(left_knee.get('at_key_moment'))}° @ {_fmt(left_knee.get('time_at_key_moment'), 1)}s)\n"
                     f"- Right Knee: {_fmt(right_knee.get('min'))}° - {_fmt(right_knee.get('max'))}° "
                     f"(deepest: {_fmt(right_knee.get('at_key_moment'))}° @ {_fmt(right_knee.get('time_at_key_moment'), 1)}s)\n")
        left_hip = pose_data.get('left_hip', {})
        right_hip = pose_data.get('right_hip', {})
        if left_hip or right_hip:
            text += (f"- Left Hip: {_fmt(left_hip.get('min'))}° - {_fmt(left_hip.get('max'))}°\n"
                     f"- Right Hip: {_fmt(right_hip.get('min'))}° - {_fmt(right_hip.get('max'))}°\n"
                     f"- Hip level difference (max): {_fmt(pose_data.get('max_hip_angle_difference', 0), 1)}°\n")
        left_shoulder = pose_data.get('left_shoulder', {})
        right_shoulder = pose_data.get('right_shoulder', {})
        if left_shoulder.get('min', 180) < 180 or right_shoulder.get('min', 180) < 180:
            text += (f"- Left Shoulder ROM: {_fmt(left_shoulder.get('min'))}° - {_fmt(left_shoulder.get('max'))}°\n"
                     f"- Right Shoulder ROM: {_fmt(right_shoulder.get('min'))}° - {_fmt(right_shoulder.get('max'))}°\n")
        back_angle = pose_data.get('back_angle', {})
        if back_angle.get('min', 180) < 180:
            text += f"- Back/Spine Angle: {_fmt(back_angle.get('min'))}° - {_fmt(back_angle.get('max'))}°\n"
        for label, key in (("Left Elbow", "left_elbow"), ("Right Elbow", "right_elbow")):
            a = pose_data.get(key, {})
            if a.get('min', 180) < 180:
                text += f"- {label}: {_fmt(a.get('min'))}° - {_fmt(a.get('max'))}°\n"
        tt = pose_data.get('trunk_tibia_difference', {})
        if tt.get('min', 180) < 180:
            text += f"- Trunk vs tibia difference: {_fmt(tt.get('min'))}° - {_fmt(tt.get('max'))}° (side views; near 0 = parallel)\n"
        gap = pose_data.get('normalized_hand_gap')
        if isinstance(gap, (int, float)):
            text += f"- Fist gap (smallest during hold): {gap:.2f} hand lengths (wrist-to-wrist over wrist-to-middle-fingertip)\n"
        if pose_data.get('exercise_detected'):
            text += (f"- Exercise position detected: count {pose_data.get('exercise_count', 0)}, "
                     f"{_fmt(pose_data.get('exercise_position_percentage', 0))}% of frames in position, "
                     f"estimated hold {_fmt(pose_data.get('estimated_hold_duration', 0), 1)}s\n")
        detection_rate = pose_data.get('person_detection_rate', 0)
        text += f"- Person detected in {_fmt(detection_rate * 100 if detection_rate else 0)}% of frames\n"
    return text


def build_exercise_prompt(
    exercise_type: str,
    exercise_name: str,
    view_types: List[str],
    pose_data_list: List[Dict],
    side: Optional[str] = None,
    pushup_variation: Optional[str] = None,
    scoring_profile: Optional[str] = None,
) -> str:
    """Build the spec-driven analysis prompt for one exercise (or one side of it)."""
    config = EXERCISE_CRITERIA.get(exercise_type, EXERCISE_CRITERIA["functional_squat"])
    view_display = ', '.join(v.replace('_', ' ').title() for v in view_types)
    tag_lines = '\n'.join(f"- {tag}: {obs} ({imp})" for tag, obs, imp in config["tags"])
    obs_format = ',\n    '.join(f'"{k}": "<plain-language observation>"' for k in config["observations_keys"])

    mode = config["mode"]
    if mode == "reps":
        mode_rules = "Report rep_scores as a list of three integers (fewer if a rep was invalidated). Leave clean_duration_seconds, valid_rep_count and normalized_hand_gap null."
    elif mode == "hold":
        mode_rules = f"Report clean_duration_seconds (0 to {config['hold_seconds']}). Leave rep_scores, valid_rep_count and normalized_hand_gap null."
    elif mode == "shoulder":
        mode_rules = "Report normalized_hand_gap (fist gap divided by hand length, 0.0 when fists touch). Leave rep_scores, clean_duration_seconds and valid_rep_count null."
    else:
        mode_rules = (f"Selected variation: {pushup_variation or 'standard'}. Selected scoring profile: {scoring_profile or 'men'}. "
                      "Report valid_rep_count and attempted_rep_count. Leave rep_scores, clean_duration_seconds and normalized_hand_gap null.")

    side_note = f"\nThis request covers the {side.upper()} side / configuration only; score it on its own.\n" if side else ""

    prompt = f"""
You are a movement analyst supporting a coach. Analyse this {config['name']} recording under the
KYNETIQ Edge AI assessment criteria (rule version {RULE_VERSION}).

## Exercise
**{exercise_name}**
Required recordings: {config['views']}
Camera views provided ({len(view_types)}): {view_display}
{side_note}
{SHARED_RULES}

## What to analyse
{config['focus']}

## Compensation tags for this exercise (use only these)
{tag_lines}

## Scoring
{config['scoring']}
{mode_rules}
{format_pose_data(view_types, pose_data_list)}

## Required response format (JSON only)
{{
  "rateable": <true|false>,
  "not_rateable_reason": <"short reason" or null>,
  "score": <0, 1, 2, 3, or null when not rateable>,
  "zero_reason": <"unable" or null>,
  "rep_scores": <[int, ...] or null>,
  "clean_duration_seconds": <number or null>,
  "valid_rep_count": <integer or null>,
  "attempted_rep_count": <integer or null>,
  "normalized_hand_gap": <number or null>,
  "compensation_tags": [
    {{
      "tag": "<TAG_FROM_LIST>",
      "severity": "<transient|mild|moderate|major>",
      "confidence": <0.00-1.00>,
      "first_detected_time": <seconds from task start>,
      "duration_seconds": <number>,
      "rep_index": <1-3 or null>,
      "observation": "<what was seen, e.g. 'Right knee moved medial to the foot line during two of three repetitions'>"
    }}
  ],
  "coach_observation": "<2-3 plain-language sentences describing what was seen. No causes, no diagnoses.>",
  "observations": {{
    {obs_format}
  }},
  "strengths": ["<what was done well>"],
  "improvements": ["<actionable, non-diagnostic suggestion>"],
  "summary": "<2-3 sentences justifying the score from the tags>"
}}

Rules for the JSON: output only the object; numbers are plain numbers; use only tags from the
list; an empty compensation_tags list is correct when nothing meaningful was seen.
"""
    return prompt


# ============================================================
# DETERMINISTIC RULES
# ============================================================

PUSHUP_BANDS = {
    # variation -> profile -> (threshold for top band, threshold for middle band)
    "standard": {"men": (15, 5), "women": (10, 3)},
    "knee": {"men": (15, 1), "women": (10, 1)},
}


def pushup_base_score(valid_reps: int, variation: str, profile: str) -> Tuple[int, Optional[str], bool]:
    """Returns (score, zero_reason, knee_retest_offered) per the Upper FMS push-up tables."""
    variation = variation if variation in PUSHUP_BANDS else "standard"
    profile = profile if profile in ("men", "women") else "men"
    top, mid = PUSHUP_BANDS[variation][profile]
    if valid_reps <= 0:
        return 0, "unable", variation == "standard"
    if variation == "knee":
        return (2 if valid_reps >= top else 1), None, False
    if valid_reps >= top:
        return 3, None, False
    if valid_reps >= mid:
        return 2, None, False
    # Below the minimum standard band: scored 1, knee retest offered after rest
    return 1, None, True


def hold_cap(seconds: Optional[float], exercise_type: str) -> Optional[int]:
    """Clean-hold duration bands for balance (20 s) and plank (30 s)."""
    if seconds is None:
        return None
    if exercise_type == "single_leg_balance":
        return 1 if seconds < 10 else (2 if seconds < 20 else 3)
    if exercise_type == "plank":
        return 1 if seconds < 20 else (2 if seconds < 30 else 3)
    return None


def hand_gap_cap(gap: Optional[float]) -> Optional[int]:
    if gap is None:
        return None
    return 3 if gap <= 1.0 else (2 if gap <= 1.5 else 1)


def severity_cap(tags: List[Dict]) -> int:
    """Lowest score the reported tags allow (3 = no cap)."""
    cap = 3
    for t in tags:
        name = t.get("tag", "")
        if name in NON_SCORING_TAGS or name in INVALID_REP_TAGS:
            continue
        sev = "major" if name in ALWAYS_MAJOR_TAGS else t.get("severity")
        if sev == "major":
            cap = min(cap, 2 if name in SECONDARY_TAGS else 1)
        elif sev == "moderate":
            cap = min(cap, 2)
    return cap


def apply_rules(
    result: Dict,
    exercise_type: str,
    reported_pain: bool = False,
    pushup_variation: Optional[str] = None,
    scoring_profile: Optional[str] = None,
) -> Dict:
    """Enforce the spec's deterministic parts on top of the model's judgement."""
    tags = [t for t in (result.get("compensation_tags") or []) if isinstance(t, dict) and t.get("tag")]
    for t in tags:
        if t["tag"] in ALWAYS_MAJOR_TAGS:
            t["severity"] = "major"
    result["compensation_tags"] = tags
    result["compensations_detected"] = [t.get("observation") or t["tag"] for t in tags]
    result["rule_version"] = RULE_VERSION
    result.setdefault("knee_retest_offered", False)

    if reported_pain:
        result.update(rateable=True, not_rateable_reason=None, score=0,
                      zero_reason="pain", classification="Pain")
        return result

    if result.get("rateable") is False:
        result.update(score=None, zero_reason=None, classification="Not Rateable",
                      not_rateable_reason=result.get("not_rateable_reason") or "recording could not be evaluated")
        return result
    result["rateable"] = True
    result["not_rateable_reason"] = None

    score = result.get("score")
    if score not in (0, 1, 2, 3):
        score = 2

    if exercise_type == "pushups" and isinstance(result.get("valid_rep_count"), (int, float)):
        score, zero_reason, retest = pushup_base_score(
            int(result["valid_rep_count"]), pushup_variation or "standard", scoring_profile or "men")
        result["zero_reason"] = zero_reason
        result["knee_retest_offered"] = retest
    elif exercise_type == "pushups" and pushup_variation == "knee":
        score = min(score, 2)

    if score > 0:
        for cap in (severity_cap(tags),
                    hold_cap(result.get("clean_duration_seconds"), exercise_type),
                    hand_gap_cap(result.get("normalized_hand_gap")) if exercise_type == "shoulder_mobility" else None):
            if cap is not None:
                score = min(score, cap)

    result["score"] = score
    if score == 0:
        if result.get("zero_reason") != "unable":
            result["zero_reason"] = "unable"
        result["classification"] = "Unable"
    else:
        result["zero_reason"] = None
        result["classification"] = {3: "Optimal", 2: "Compensated", 1: "Dysfunctional"}[score]
    return result


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_exercise(
    exercise_type: str,
    exercise_name: str,
    videos_base64: List[str],
    view_types: List[str],
    pose_data_list: List[Dict],
    reported_pain: bool = False,
    side: Optional[str] = None,
    mime_type: str = "video/mp4",
    pushup_variation: Optional[str] = None,
    scoring_profile: Optional[str] = None,
) -> Dict:
    """Analyze exercise video(s) using Gemini with native video understanding."""
    response_text = ""
    try:
        prompt = build_exercise_prompt(
            exercise_type, exercise_name, view_types, pose_data_list,
            side, pushup_variation, scoring_profile,
        )
        if reported_pain:
            prompt += "\n\nATHLETE REPORTED PAIN during this test. The score is 0-P regardless of movement quality; still report the tags you see."

        contents = [prompt]
        for i, video_b64 in enumerate(videos_base64):
            view_label = view_types[i] if i < len(view_types) else f"view_{i+1}"
            contents.append(f"\n[Video {i+1}: {view_label.replace('_', ' ').title()}]")
            contents.append(genai.types.Part.from_bytes(data=base64.b64decode(video_b64), mime_type=mime_type))

        print(f"Sending {len(videos_base64)} video(s) to Gemini for analysis")
        response = client.models.generate_content(model=MODEL, contents=contents)
        response_text = response.text.strip()

        if '```' in response_text:
            json_lines, in_json = [], False
            for line in response_text.split('\n'):
                if '```json' in line:
                    in_json = True
                    continue
                if '```' in line and in_json:
                    in_json = False
                    continue
                if in_json:
                    json_lines.append(line)
            if json_lines:
                response_text = '\n'.join(json_lines)
        if not response_text.startswith('{'):
            start_idx, end_idx = response_text.find('{'), response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx + 1]

        analysis_result = apply_rules(
            json.loads(response_text), exercise_type, reported_pain, pushup_variation, scoring_profile)
        analysis_result["model"] = MODEL
        return {"success": True, "analysis": analysis_result}

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Failed to parse AI response: {str(e)}",
            "raw_response": response_text[:1000] if response_text else None
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


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
            "service": "fms_analysis_api",
            "version": "5.1-sensor-data",
            "rule_version": RULE_VERSION,
            "supported_exercises": list(EXERCISE_CRITERIA.keys()),
            "timestamp": datetime.now().isoformat()
        })
    
    @app.get("/test-connection")
    def test_connection():
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents="Say 'connected' and nothing else."
            )
            return jsonify({
                "success": True,
                "connected": True,
                "model": MODEL,
                "response": response.text.strip(),
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "connected": False,
                "model": MODEL,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.post("/analyze")
    def analyze():
        """
        Multi-exercise analysis endpoint.
        
        Expected JSON body:
        {
            "exercise_type": "single_leg_balance",
            "exercise_name": "Single Leg Balance",
            "category": "lower_fms",
            "videos": ["<base64>", "<base64>", "<base64>", "<base64>"],
            "view_types": ["front_left", "left_side", "front_right", "right_side"],
            "mime_type": "video/mp4",
            "reported_pain": false,
            "pose_detection_data": [
                { view 1 data },
                { view 2 data },
                { view 3 data },
                { view 4 data }
            ],
            "pushup_variation": "standard" | "knee",
            "scoring_profile": "men" | "women"
        }
        """
        try:
            data = request.get_json(force=True) or {}
            
            # Handle both single video and multi-video formats
            if "videos" in data:
                videos = data["videos"]
            elif "video" in data:
                videos = [data["video"]]
            else:
                return jsonify({
                    "success": False,
                    "error": "videos or video (base64) is required",
                    "timestamp": datetime.now().isoformat()
                }), 400
            
            exercise_type = data.get("exercise_type", "functional_squat")
            exercise_name = data.get("exercise_name", "Unknown Exercise")
            view_types = data.get("view_types", ["front"])
            reported_pain = data.get("reported_pain", False)
            pose_data_list = data.get("pose_detection_data", [])
            side = data.get("side")  # "left" or "right" for bilateral exercises
            pushup_variation = data.get("pushup_variation")  # "standard" | "knee", selected by the athlete
            scoring_profile = data.get("scoring_profile")    # "men" | "women", selected by the athlete

            # Ensure pose_data_list is a list
            if isinstance(pose_data_list, dict):
                pose_data_list = [pose_data_list]

            print(f"=== Analyze Request ===")
            print(f"  Exercise: {exercise_name} ({exercise_type})")
            print(f"  Videos: {len(videos)}")
            print(f"  Views: {view_types}")
            print(f"  Side: {side or 'both'}")
            print(f"  Pain reported: {reported_pain}")

            result = analyze_exercise(
                exercise_type=exercise_type,
                exercise_name=exercise_name,
                videos_base64=videos,
                view_types=view_types,
                pose_data_list=pose_data_list,
                reported_pain=reported_pain,
                side=side,
                pushup_variation=pushup_variation,
                scoring_profile=scoring_profile,
            )
            
            if result["success"]:
                response_data = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "result": result["analysis"],
                    "exercise_type": exercise_type,
                    "rule_version": RULE_VERSION,
                }
                if side:
                    response_data["side"] = side
                return jsonify(response_data)
            else:
                return jsonify({
                    "success": False,
                    "error": result.get("error", "Analysis failed"),
                    "raw_response": result.get("raw_response"),
                    "timestamp": datetime.now().isoformat()
                }), 500
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }), 500
    
    @app.get("/exercises")
    def list_exercises():
        """List all supported exercises."""
        return jsonify({
            "lower_fms": [
                {"id": "functional_squat", "name": "Functional Squat", "views": 2, "mode": "rep_based", "reps": 3},
                {"id": "single_leg_balance", "name": "Single Leg Balance", "views": 4, "mode": "timed"},
                {"id": "inline_lunge", "name": "In-Line Lunge", "views": 4, "mode": "rep_based", "reps": 3}
            ],
            "upper_fms": [
                {"id": "plank", "name": "Plank", "views": 1, "mode": "timed"},
                {"id": "shoulder_mobility", "name": "Shoulder Mobility (IR/ER)", "views": 2, "mode": "timed"},
                {"id": "pushups", "name": "Push-Ups", "views": 1, "mode": "timed"}
            ]
        })

    @app.get("/")
    def root():
        return jsonify({
            "service": "KynetiqEdge FMS Analysis API",
            "version": "3.3-updated",
            "model": MODEL,
            "endpoints": {
                "GET /": "API info",
                "GET /health": "Health check",
                "GET /test-connection": "Test OpenAI connection",
                "GET /exercises": "List supported exercises",
                "POST /analyze": "Analyze exercise video(s)"
            },
            "supported_exercises": {
                "lower_fms": [
                    "functional_squat (2 views: front, side) - 3 reps",
                    "single_leg_balance (4 views: front L, side L, front R, side R)",
                    "inline_lunge (4 views: front L, side L, front R, side R) - 3 reps"
                ],
                "upper_fms": [
                    "plank (1 view: side)",
                    "shoulder_mobility (2 views: back L arm up, back R arm up)",
                    "pushups (1 view: left side)"
                ]
            }
        })
    
    return app


app = create_app()

if __name__ == "__main__":
    print("Starting FMS Analysis API v3.1...")
    print(f"Model: {MODEL}")
    print(f"Supported exercises: {list(EXERCISE_CRITERIA.keys())}")
    app.run(host="0.0.0.0", port=8000, debug=True)