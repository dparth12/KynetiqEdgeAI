"""Self-check for the deterministic scoring rules. Run: python3 api/test_rules.py"""
import sys, types, os
sys.path.insert(0, os.path.dirname(__file__))
# Stub the Gemini SDK so the module imports without the dependency or a key
google = types.ModuleType("google"); genai = types.ModuleType("google.genai")
genai.Client = lambda **kw: None; genai.types = types.SimpleNamespace(Part=None)
google.genai = genai; sys.modules["google"] = google; sys.modules["google.genai"] = genai
flask = types.ModuleType("flask"); flask.Flask = lambda *a, **k: types.SimpleNamespace(get=lambda *a, **k: (lambda f: f), post=lambda *a, **k: (lambda f: f))
flask.request = None; flask.jsonify = lambda *a, **k: None; sys.modules["flask"] = flask
flask_cors = types.ModuleType("flask_cors"); flask_cors.CORS = lambda app: None; sys.modules["flask_cors"] = flask_cors

import index as m

def tag(name, sev="mild", rep=None):
    return {"tag": name, "severity": sev, "confidence": 0.9, "first_detected_time": 1.0, "duration_seconds": 1.0, "rep_index": rep, "observation": name.lower()}

# Push-up base scoring tables (Upper FMS v1.0, QA examples)
assert m.pushup_base_score(18, "standard", "men") == (3, None, False)
assert m.pushup_base_score(12, "standard", "women") == (3, None, False)
assert m.pushup_base_score(14, "knee", "women") == (2, None, False)
assert m.pushup_base_score(8, "knee", "men") == (1, None, False)
assert m.pushup_base_score(4, "standard", "men") == (1, None, True)      # below band -> knee retest offered
assert m.pushup_base_score(0, "knee", "women") == (0, "unable", False)

# Quality cap: 18 valid standard reps with persistent moderate hip sag -> 2
r = m.apply_rules({"score": 3, "valid_rep_count": 18, "compensation_tags": [tag("PU_LUMBAR_EXTENSION", "moderate")]},
                  "pushups", pushup_variation="standard", scoring_profile="men")
assert r["score"] == 2 and r["classification"] == "Compensated", r
# Invalid-rep tags never cap on their own; they only shrink the valid count the model reports
r = m.apply_rules({"score": 3, "valid_rep_count": 16, "compensation_tags": [tag("PU_INCOMPLETE_LOCKOUT", "major", 5)]},
                  "pushups", pushup_variation="standard", scoring_profile="men")
assert r["score"] == 3, r

# Severity ladder: major caps at 1, moderate at 2, transient/mild leave the model's score
assert m.apply_rules({"score": 3, "compensation_tags": [tag("SQUAT_HEEL_RISE", "major")]}, "functional_squat")["score"] == 1
assert m.apply_rules({"score": 3, "compensation_tags": [tag("SQUAT_KNEE_MEDIAL", "moderate")]}, "functional_squat")["score"] == 2
assert m.apply_rules({"score": 3, "compensation_tags": [tag("SQUAT_KNEE_MEDIAL", "transient")]}, "functional_squat")["score"] == 3
# Immediate major events are forced to major whatever the model said
r = m.apply_rules({"score": 3, "compensation_tags": [tag("SLB_TOUCHDOWN", "mild")]}, "single_leg_balance")
assert r["score"] == 1 and r["compensation_tags"][0]["severity"] == "major"
# Secondary tags cap at most 2; review-only tags never cap
assert m.apply_rules({"score": 3, "compensation_tags": [tag("PLANK_FORWARD_HEAD", "major")]}, "plank")["score"] == 2
assert m.apply_rules({"score": 3, "compensation_tags": [tag("SM_SCAPULAR_BORDER_PROMINENCE", "major")]}, "shoulder_mobility")["score"] == 3

# Clean-hold bands
assert m.apply_rules({"score": 3, "clean_duration_seconds": 22}, "plank")["score"] == 2
assert m.apply_rules({"score": 3, "clean_duration_seconds": 15}, "plank")["score"] == 1
assert m.apply_rules({"score": 3, "clean_duration_seconds": 20}, "single_leg_balance")["score"] == 3
assert m.apply_rules({"score": 3, "clean_duration_seconds": 9.5}, "single_leg_balance")["score"] == 1
# Shoulder hand-gap bands
assert m.apply_rules({"score": 3, "normalized_hand_gap": 1.0}, "shoulder_mobility")["score"] == 3
assert m.apply_rules({"score": 3, "normalized_hand_gap": 1.3}, "shoulder_mobility")["score"] == 2
assert m.apply_rules({"score": 3, "normalized_hand_gap": 1.6}, "shoulder_mobility")["score"] == 1

# Pain from the athlete wins over everything and keeps the tags
r = m.apply_rules({"score": 3, "compensation_tags": [tag("SQUAT_HEEL_RISE")]}, "functional_squat", reported_pain=True)
assert r["score"] == 0 and r["zero_reason"] == "pain" and r["rateable"] and len(r["compensation_tags"]) == 1
# Not rateable -> score null, never a 0
r = m.apply_rules({"rateable": False, "not_rateable_reason": "sagittal side view required", "score": 1}, "plank")
assert r["score"] is None and r["classification"] == "Not Rateable"
# Model zero without pain is "unable"
r = m.apply_rules({"score": 0}, "inline_lunge")
assert r["zero_reason"] == "unable" and r["classification"] == "Unable"
# Every prompt builds and names only its own tags
for ex, cfg in m.EXERCISE_CRITERIA.items():
    p = m.build_exercise_prompt(ex, cfg["name"], ["front"], [], None, "knee", "women")
    assert all(t in p for t, _, _ in cfg["tags"]) and m.RULE_VERSION in p
print("rules ok:", m.RULE_VERSION)
# Sensor block prints elbow and trunk/tibia only when measured
txt = m.format_pose_data(["left_side"], [{"left_elbow": {"min": 70, "max": 160}, "trunk_tibia_difference": {"min": 4, "max": 12}}])
assert "Left Elbow: 70" in txt and "Trunk vs tibia difference: 4" in txt
assert "Elbow" not in m.format_pose_data(["front"], [{}]) and "tibia" not in m.format_pose_data(["front"], [{}])
print("sensor block ok")
# Sensor fist gap in hand lengths, omitted when absent
assert "1.20 hand lengths" in m.format_pose_data(["back_left_arm_up"], [{"normalized_hand_gap": 1.2}])
assert "hand lengths" not in m.format_pose_data(["back_left_arm_up"], [{"normalized_hand_gap": None}])
assert "fist gap in hand lengths" in m.build_exercise_prompt("shoulder_mobility", "Shoulder Mobility", ["back_left_arm_up"], [], None, None, None)
print("hand gap ok")
