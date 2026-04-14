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
# MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

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
import google.generativeai as genai
import json
from datetime import datetime
from typing import Dict, Optional, List, Tuple

# ============================================================
# CONFIG & SETUP
# ============================================================
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


# ============================================================
# EXERCISE-SPECIFIC PROMPTS
# ============================================================

EXERCISE_PROMPTS = {
    "functional_squat": {
        "name": "Functional Squat",
        "category": "lower_fms",
        "focus_areas": """
Analyze the overhead deep squat from FRONT and SIDE views:

## USING QUICKPOSE SENSOR DATA
The pose detection provides ACCURATE knee angle measurements:
- **left_knee** and **right_knee** angles show depth achieved
- If min knee angle <90° = excellent depth (below parallel)
- If min knee angle 90-100° = good depth (near parallel)
- If min knee angle >100° = insufficient depth
- TRUST this sensor data for depth assessment

YOUR visual analysis should focus on what sensors CAN'T capture:
1. **Torso Position** - Torso and tibia should be parallel (minimal forward lean)
   - Measure forward lean angle from vertical
2. **Heel Contact** - CRITICAL: Heels must stay FLAT on ground throughout
   - Any heel lift = compensation
3. **Knee Tracking** - Watch from FRONT view for:
   - Valgus (knees collapsing inward)
   - Varus (knees bowing outward)
   - Knees should track over 2nd/3rd toe
4. **Arm/Overhead Position** - Arms/dowel should stay overhead without dropping forward
5. **Balance & Control** - Smooth, controlled descent and ascent

FROM SIDE VIEW: Check torso angle, heel contact
FROM FRONT VIEW: Check knee tracking (valgus/varus), balance
Use the SENSOR DATA for depth assessment, your visual analysis for everything else.
""",
        "scoring": """
**Score 3 (Optimal):**
- Thighs go BELOW parallel (hip crease below knee)
- Torso and tibia parallel (minimal forward lean <10°)
- Heels stay FLAT on ground throughout
- Knees aligned over feet (no valgus/varus)
- Arms/dowel maintained overhead, no forward lean

**Score 2 (Compensated):**
- Reaches parallel but with ONE OR MORE of:
  - Heels lift off ground
  - Torso leans forward >10-15° from vertical
  - Mild knee valgus or varus (<10°)
  - Arms drop forward slightly

**Score 1 (Dysfunctional):**
- Cannot reach parallel depth
- Major loss of balance
- Severe forward lean (>20-30°)
- Significant knee valgus/varus (>10°)
- Multiple compensations present

**Score 0 (Pain):**
- Reports pain during movement
""",
        "observations_keys": ["depth", "torso", "heels", "knees", "arms", "overall"]
    },
    
    "single_leg_balance": {
        "name": "Single Leg Balance (Both Legs)",
        "category": "lower_fms",
        "focus_areas": """
Analyze BOTH legs - you will see 4 views:
1. Front view - Left leg standing
2. Left side view - Left leg standing
3. Front view - Right leg standing
4. Right side view - Right leg standing

## USING QUICKPOSE SENSOR DATA
The pose detection provides ACCURATE measurements:
- **hip_drop_detected**: TRUE/FALSE based on >10° difference between left/right hip angles
- **max_hip_angle_difference**: The maximum hip angle asymmetry detected (degrees)
- **estimated_hold_duration**: Time in exercise position based on frame analysis
- **left_hip** and **right_hip** angles throughout the movement

TRUST THIS DATA for hip drop assessment. If hip_drop_detected is TRUE and max_hip_angle_difference >10°,
the person has significant pelvic drop indicating glute medius weakness.

YOUR visual analysis should confirm and add context:
1. **Stability** - Amount of sway, wobble, or corrections needed
2. **Trunk position** - Should be upright, not leaning to compensate
3. **Compensation strategies** - Arm waving, trunk rotation, excessive ankle movement
4. **Touch downs** - Did foot touch the ground?

DURATION ASSESSMENT:
- Use the **estimated_hold_duration** from sensor data as a baseline
- Your visual analysis should confirm this and note when balance is LOST

IMPORTANT: Compare LEFT vs RIGHT leg performance and note any asymmetries.
""",
        "scoring": """
**Score 3 (Optimal):**
- Holds ≥20 seconds on EACH leg with minimal sway
- Trunk upright on both sides
- NO hip drop >10° on either side (pelvis stays level)
- Symmetrical performance between legs

**Score 2 (Compensated):**
- Holds 10-19 seconds on one or both legs
- Mild sway or compensation
- Minor hip drop (5-10°) acceptable
- Minor asymmetry between sides

**Score 1 (Dysfunctional):**
- Holds <10 seconds on one or both legs
- Frequent wobble or corrections
- Touches down or loses balance
- Significant hip drop (>10°)
- Significant asymmetry between legs

**Score 0 (Pain):**
- Pain reported

ASYMMETRY FLAG: If one leg holds significantly longer (>5 sec difference) or shows notably worse hip drop, flag this in your assessment.
""",
        "observations_keys": ["stability", "hipDrop", "trunkPosition", "duration", "symmetry", "leftLeg", "rightLeg", "overall"]
    },
    
    "inline_lunge": {
        "name": "In-Line Lunge (Both Legs)",
        "category": "lower_fms",
        "focus_areas": """
Analyze BOTH legs - you will see 4 views:
1. Front view - Left leg forward
2. Left side view - Left leg forward
3. Front view - Right leg forward
4. Right side view - Right leg forward

## USING QUICKPOSE SENSOR DATA
The pose detection provides ACCURATE measurements:
- **left_knee** and **right_knee**: Angle tracking during lunge
- **left_hip** and **right_hip**: Hip angles for balance assessment
- Use knee angle data to assess depth and tracking

For each leg forward, CAREFULLY assess:
1. **Trunk position** - MUST be upright throughout the movement
2. **Front heel contact** - MUST stay flat on ground (heel lift = compensation)
3. **Back knee position** - Should touch floor JUST BEHIND the front heel (not beside it)
4. **Knee tracking** - Use knee angle data to detect valgus (inward) or varus (outward)
5. **Balance & control** - No wobble, no stepping off the line
6. **Line alignment** - Feet must stay on the imaginary line (heel-to-toe alignment)

CRITICAL CHECKPOINTS:
- From SIDE VIEW: Check trunk angle (should be vertical), front heel contact, back knee position
- From FRONT VIEW: Check knee tracking (valgus/varus), balance, line alignment

IMPORTANT: Compare LEFT vs RIGHT leg forward performance and note any asymmetries.
""",
        "scoring": """
**Score 3 (Optimal):**
- Trunk upright throughout on BOTH sides
- Front heel stays FLAT on both legs
- Back knee touches floor JUST BEHIND front heel
- NO wobble on either side
- Front knee tracks directly over foot (no valgus/varus)
- Symmetrical performance between legs

**Score 2 (Compensated):**
- Minor trunk lean (<15°) on one or both sides
- Slight heel lift
- Mild valgus/varus drift (<10°)
- Minor loss of contact with dowel (if used)
- Minor asymmetry between legs

**Score 1 (Dysfunctional):**
- Cannot complete lunge under control
- Loss of balance on one or both sides
- Major trunk lean (>15°)
- Steps off the line
- Significant valgus/varus (>10°)
- Significant asymmetry between legs

**Score 0 (Pain):**
- Reports pain during lunge

ASYMMETRY FLAG: Note if one side shows significantly worse performance (more compensation, less control).
""",
        "observations_keys": ["alignment", "kneeTracking", "balance", "trunkPosition", "symmetry", "leftLeg", "rightLeg", "overall"]
    },
    
    "plank": {
        "name": "Plank",
        "category": "upper_fms",
        "focus_areas": """
Analyze the plank hold from SIDE VIEW:

## FOCUS: CONSISTENT HOLD
The main goal is to assess if the person can HOLD a plank position consistently.
Minor adjustments and small movements are NORMAL and should NOT be penalized.

## WHAT TO LOOK FOR
1. **Consistent hold** - Did they maintain the position throughout?
2. **General body line** - Roughly straight from head to heels (doesn't need to be perfect)
3. **Major breakdowns** - Did they collapse, drop to knees, or completely lose position?

## WHAT TO IGNORE (Don't penalize)
- Small adjustments or shifts
- Minor trembling (this is normal under load)
- Slight head movements
- Small hip position changes
- Not having "perfect" form

## DURATION ASSESSMENT
- The video shows the ENTIRE recording (30 seconds)
- If they maintain plank throughout = full 30 seconds
- Only note duration if they visibly break/collapse mid-way
- Trust the video over sensor data for duration
""",
        "scoring": """
**Score 3 (Optimal):**
- Holds consistently for the FULL 30 seconds
- Maintains general straight body line
- Minor adjustments are fine
- No major collapse or hip drop to the floor

**Score 2 (Acceptable):**
- Holds 20-29 seconds before needing to stop
- OR holds full duration with noticeable but acceptable form variations
- Some hip sag or pike but still maintaining position
- Generally holds the position

**Score 1 (Dysfunctional):**
- Holds <20 seconds before breaking position
- Significant hip collapse (hips nearly touching ground)
- Cannot maintain any consistent hold
- Has to restart or take breaks

**Score 0 (Pain):**
- Pain reported
- Cannot get into plank position at all

BE GENEROUS: If the person is clearly trying and maintaining a plank position for the full time, even with imperfect form, lean toward Score 3 or 2.
""",
        "observations_keys": ["spineAlignment", "hipPosition", "holdTime", "stability", "overall"]
    },
    
    "shoulder_mobility": {
        "name": "Shoulder Mobility (IR/ER)",
        "category": "upper_fms",
        "focus_areas": """
Analyze shoulder internal/external rotation from BACK VIEW:

## USING QUICKPOSE SENSOR DATA
The pose detection provides ACCURATE shoulder angle measurements:
- **left_shoulder**: ROM range (min and max angles during movement)
- **right_shoulder**: ROM range (min and max angles during movement)

These angles help quantify the range of motion achieved.
TRUST this data for ROM assessment.

YOUR visual analysis should assess:
1. **Fist distance** - How close the fists get to each other
2. **Hand position** - Top hand reaching over shoulder, bottom hand reaching behind back
3. **Compensations** - Trunk rotation, shoulder hiking, elbow position
4. **Symmetry** - Compare if both sides tested
5. **Range achieved** - Estimate in hand-lengths apart

Use the shoulder angle data to inform your assessment of ROM achieved.
""",
        "scoring": """
**Score 3 (Optimal):**
- Fists touch (within 0 hand lengths)

**Score 2 (Acceptable):**
- Fists within 1 hand length (but not touching)

**Score 1 (Dysfunctional):**
- Fists farther apart than 1.5 hand lengths

**Score 0 (Pain):**
- Pain during attempt
""",
        "observations_keys": ["fistDistance", "shoulderRange", "symmetry", "overall"]
    },
    
    "pushups": {
        "name": "Push-Ups",
        "category": "upper_fms",
        "focus_areas": """
Analyze push-ups from SIDE VIEW:

## FOCUS: COUNT REPS
The main goal is to COUNT how many push-ups the person completes.
Side view lets you see depth and body position clearly.

## WHAT TO COUNT
1. **Rep count** - Count push-ups where chest lowers and arms extend
2. **Type** - Note if full push-ups (toes) or knee push-ups (knees on ground)

## FORM ASSESSMENT (Secondary)
- Look for generally straight body line (neutral trunk)
- Don't be too strict - minor form variations are acceptable
- Focus more on REP COUNT than perfect form

## COUNTING RULES
- Count any rep where chest visibly lowers toward ground
- Count any rep where arms extend back up
- Be GENEROUS with counting - if it looks like a push-up, count it
- Note if they switch from full to knee push-ups mid-way

User will tap STOP when done - count all reps in the video.
""",
        "scoring": """
**Score 3 - Full Push-Ups (Good Control):**
- ≥15 reps (men) / ≥10 reps (women)
- Full push-ups (on toes)
- Neutral trunk generally maintained

**Score 2 - Full Push-Ups (Limited Volume):**
- 5-14 reps (men) / 3-9 reps (women)
- Full push-ups performed
- Form doesn't need to be perfect

**Score 1 - Knee Push-Ups Only:**
- ≥5 knee push-ups with reasonable control
- Cannot perform full push-ups
- Or fewer than 5 full push-ups

**Score 0 - Unable / Pain:**
- <5 knee push-ups total
- Pain reported
- Cannot perform push-ups

DEFAULT: If gender unknown, use male thresholds.
BE GENEROUS with rep counting - focus on effort and reps completed.
""",
        "observations_keys": ["repCount", "bodyPosition", "depth", "control", "overall"]
    }
    # Step down tests removed per feedback
}


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_exercise_prompt(
    exercise_type: str,
    exercise_name: str,
    view_types: List[str],
    pose_data_list: List[Dict],
    scoring_criteria: Optional[Dict] = None,
    side: Optional[str] = None
) -> str:
    """Build exercise-specific analysis prompt."""

    exercise_config = EXERCISE_PROMPTS.get(exercise_type, EXERCISE_PROMPTS["functional_squat"])

    # If analyzing just one side, adjust the focus areas
    if side and exercise_type in ["single_leg_balance", "inline_lunge"]:
        side_upper = side.upper()
        focus_override = f"""
Analyze the {side_upper} SIDE ONLY - you will see 2 views:
1. Front view - {side.capitalize()} leg {'standing' if exercise_type == 'single_leg_balance' else 'forward'}
2. {side.capitalize()} side view - {side.capitalize()} leg {'standing' if exercise_type == 'single_leg_balance' else 'forward'}

For the {side_upper} side, assess:
"""
        if exercise_type == "single_leg_balance":
            focus_override += """
1. **Stability** - Amount of sway, wobble, or corrections
2. **Hip drop** - Pelvis staying level vs dropping on unsupported side
3. **Trunk position** - Upright vs leaning
4. **Duration quality** - Ability to maintain position
5. **Compensation strategies** - Arm waving, trunk rotation, excessive ankle movement
"""
        else:  # inline_lunge
            focus_override += """
1. **Trunk position** - Upright vs leaning
2. **Front heel contact** - Stays flat or lifts
3. **Knee tracking** - Front knee over foot, no valgus/varus
4. **Back knee position** - Touches behind front heel
5. **Balance & control** - Stability throughout, no stepping off line
6. **Alignment** - Feet stay on the line
"""
        exercise_config = dict(exercise_config)
        exercise_config["focus_areas"] = focus_override
    
    # Format view types for display
    view_display = []
    for v in view_types:
        formatted = v.replace('_', ' ').title()
        view_display.append(formatted)
    
    prompt = f"""
You are an expert movement analyst and physical therapist specializing in Functional Movement Screening (FMS).
Analyze this {exercise_config['name']} assessment and provide a detailed evaluation.

## Exercise Being Assessed
**{exercise_name}**

## Camera Views Provided ({len(view_types)} views)
{', '.join(view_display)}

## Your Analysis Focus
{exercise_config['focus_areas']}

## Scoring Criteria
{exercise_config['scoring'] if not scoring_criteria else format_custom_scoring(scoring_criteria)}
"""

    # Helper function to safely format numbers
    def fmt(val, decimals=0):
        if val is None or val == 'N/A':
            return 'N/A'
        try:
            if decimals == 0:
                return f"{float(val):.0f}"
            else:
                return f"{float(val):.{decimals}f}"
        except (ValueError, TypeError):
            return 'N/A'

    # Add pose detection data if available
    if pose_data_list:
        prompt += "\n## POSE DETECTION DATA (from QuickPose SDK)\n"
        prompt += "⚠️ TRUST THIS DATA - it is measured by sensors during recording.\n"
        for i, pose_data in enumerate(pose_data_list):
            view_name = view_types[i] if i < len(view_types) else f"View {i+1}"
            prompt += f"""
**{view_name.replace('_', ' ').title()}:**
"""
            # === Bilateral Knee Angles ===
            left_knee = pose_data.get('left_knee', {})
            right_knee = pose_data.get('right_knee', {})
            if left_knee or right_knee:
                prompt += f"""- Left Knee: {fmt(left_knee.get('min'))}° - {fmt(left_knee.get('max'))}° (deepest: {fmt(left_knee.get('at_key_moment'))}° @ {fmt(left_knee.get('time_at_key_moment'), 1)}s)
- Right Knee: {fmt(right_knee.get('min'))}° - {fmt(right_knee.get('max'))}° (deepest: {fmt(right_knee.get('at_key_moment'))}° @ {fmt(right_knee.get('time_at_key_moment'), 1)}s)
"""
            else:
                # Fallback to legacy fields
                prompt += f"""- Min knee angle: {pose_data.get('min_knee_angle', 'N/A')}°
- Max knee angle: {pose_data.get('max_knee_angle', 'N/A')}°
"""

            # === Hip Angles (for hip drop detection) ===
            left_hip = pose_data.get('left_hip', {})
            right_hip = pose_data.get('right_hip', {})
            if left_hip or right_hip:
                prompt += f"""- Left Hip: {fmt(left_hip.get('min'))}° - {fmt(left_hip.get('max'))}°
- Right Hip: {fmt(right_hip.get('min'))}° - {fmt(right_hip.get('max'))}°
- **HIP DROP DETECTED: {'✅ YES' if pose_data.get('hip_drop_detected') else '❌ NO'}** (max diff: {fmt(pose_data.get('max_hip_angle_difference', 0), 1)}°)
"""

            # === Shoulder Angles ===
            left_shoulder = pose_data.get('left_shoulder', {})
            right_shoulder = pose_data.get('right_shoulder', {})
            if left_shoulder.get('min', 180) < 180 or right_shoulder.get('min', 180) < 180:
                prompt += f"""- Left Shoulder ROM: {fmt(left_shoulder.get('min'))}° - {fmt(left_shoulder.get('max'))}°
- Right Shoulder ROM: {fmt(right_shoulder.get('min'))}° - {fmt(right_shoulder.get('max'))}°
"""

            # === Back Angle (for plank) ===
            back_angle = pose_data.get('back_angle', {})
            if back_angle.get('min', 180) < 180:
                prompt += f"""- Back/Spine Angle: {fmt(back_angle.get('min'))}° - {fmt(back_angle.get('max'))}° (current: {fmt(back_angle.get('current'))}°)
"""

            # === Exercise Detection ===
            if pose_data.get('exercise_detected'):
                prompt += f"""- **EXERCISE DETECTED: ✅ YES** (count: {pose_data.get('exercise_count', 0)}, {fmt(pose_data.get('exercise_position_percentage', 0))}% of frames in position)
- **ESTIMATED HOLD DURATION: {fmt(pose_data.get('estimated_hold_duration', 0), 1)}s**
"""

            # === General Data ===
            detection_rate = pose_data.get('person_detection_rate', 0)
            prompt += f"""- Person detection: {fmt(detection_rate * 100 if detection_rate else 0)}%
- Key moment at: {pose_data.get('time_at_deepest_point_seconds', 'N/A')}s
"""

    # Build observations keys
    obs_keys = exercise_config.get('observations_keys', ['overall'])
    obs_format = ',\n    '.join([f'"{k}": "<your assessment>"' for k in obs_keys])
    
    prompt += f"""

## Required Response Format (JSON)

{{
  "score": <0-3>,
  "classification": "<Optimal|Compensated|Acceptable|Dysfunctional|Pain>",
  "observations": {{
    {obs_format}
  }},
  "compensations_detected": [
    "<specific compensation observed>",
    "<another if any>"
  ],
  "strengths": [
    "<what was done well>"
  ],
  "improvements": [
    "<actionable suggestion>"
  ],
  "mobility_focus_areas": [
    "<body area needing work>"
  ],
  "exercise_specific_data": {{
    "touch_count": <integer or null>,
    "hold_duration": <number in seconds or null>,
    "fist_distance_hand_lengths": <number or null>
  }},
  "summary": "<2-3 sentences summarizing the assessment and justifying the score>"
}}

## CRITICAL RULES
1. Score based on the specific criteria for {exercise_config['name']}
2. Be specific about what you observe in each view
3. If multiple views provided, synthesize observations from all views
4. For bilateral exercises (balance, lunge), comment on symmetry between left and right
5. Output ONLY valid JSON
6. All numeric values in exercise_specific_data must be plain numbers (e.g., 0.6 not "0.6s")

Respond with ONLY the JSON object.
"""
    
    return prompt


def format_custom_scoring(scoring_criteria: Dict) -> str:
    """Format custom scoring criteria from request."""
    result = ""
    for score, criteria in [("3", "score_3"), ("2", "score_2"), ("1", "score_1"), ("0", "score_0")]:
        if criteria in scoring_criteria:
            result += f"\n**Score {score}:**\n"
            for c in scoring_criteria[criteria]:
                result += f"- {c}\n"
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
    scoring_criteria: Optional[Dict] = None,
    side: Optional[str] = None,
    mime_type: str = "video/mp4"
) -> Dict:
    """Analyze exercise video(s) using Gemini with native video understanding."""
    try:
        model = genai.GenerativeModel(MODEL)

        # Build prompt
        prompt = build_exercise_prompt(
            exercise_type,
            exercise_name,
            view_types,
            pose_data_list,
            scoring_criteria,
            side
        )

        if reported_pain:
            prompt += "\n\n⚠️ USER REPORTED PAIN: Score MUST be 0 regardless of movement quality."

        # Build content with videos directly (no frame extraction needed)
        content = [prompt]

        for i, video_b64 in enumerate(videos_base64):
            view_label = view_types[i] if i < len(view_types) else f"view_{i+1}"
            content.append(f"\n[Video {i+1}: {view_label.replace('_', ' ').title()}]")
            content.append({
                "mime_type": mime_type,
                "data": video_b64
            })

        print(f"Sending {len(videos_base64)} video(s) to Gemini for analysis")

        # Call Gemini with native video understanding
        response = model.generate_content(content)
        response_text = response.text.strip()

        # Clean up response (extract JSON from markdown if needed)
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

        if not response_text.startswith('{'):
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx + 1]

        # Parse response
        analysis_result = json.loads(response_text)

        # Validate score
        if analysis_result.get("score") not in [0, 1, 2, 3]:
            analysis_result["score"] = 2

        return {
            "success": True,
            "analysis": analysis_result
        }

    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error": f"Failed to parse AI response: {str(e)}",
            "raw_response": response_text[:1000] if 'response_text' in locals() else None
        }
    except Exception as e:
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
            "service": "fms_analysis_api",
            "version": "4.0-gemini-native-video",
            "supported_exercises": list(EXERCISE_PROMPTS.keys()),
            "timestamp": datetime.now().isoformat()
        })
    
    @app.get("/test-connection")
    def test_connection():
        try:
            model = genai.GenerativeModel(MODEL)
            response = model.generate_content("Say 'connected' and nothing else.")
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
            "scoring_criteria": { optional custom criteria }
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
            scoring_criteria = data.get("scoring_criteria")
            side = data.get("side")  # "left" or "right" for bilateral exercises

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
                scoring_criteria=scoring_criteria,
                side=side
            )
            
            if result["success"]:
                response_data = {
                    "success": True,
                    "timestamp": datetime.now().isoformat(),
                    "result": result["analysis"],
                    "exercise_type": exercise_type
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
    print(f"Supported exercises: {list(EXERCISE_PROMPTS.keys())}")
    app.run(host="0.0.0.0", port=8000, debug=True)