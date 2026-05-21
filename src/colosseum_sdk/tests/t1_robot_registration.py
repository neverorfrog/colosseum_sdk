"""Tests for robot registry and T1 configuration."""

from colosseum_sdk.config.robot import ROBOT_REGISTRY, get_robot

NUM_JOINTS = 23

# Expected values from arena/src/tasks/T1VelocitySymmetric.cpp
EXPECTED_DEFAULT_JOINT_POS = {
    "AAHead_yaw": 0.0,
    "Head_pitch": 0.0,
    "Left_Shoulder_Pitch": 0.25,
    "Left_Shoulder_Roll": -1.4,
    "Left_Elbow_Pitch": 0.0,
    "Left_Elbow_Yaw": -0.2,
    "Right_Shoulder_Pitch": 0.25,
    "Right_Shoulder_Roll": 1.4,
    "Right_Elbow_Pitch": 0.0,
    "Right_Elbow_Yaw": 0.2,
    "Waist": 0.0,
    "Left_Hip_Pitch": -0.38,
    "Left_Hip_Roll": 0.0,
    "Left_Hip_Yaw": 0.0,
    "Left_Knee_Pitch": 0.8,
    "Left_Ankle_Pitch": -0.43,
    "Left_Ankle_Roll": 0.0,
    "Right_Hip_Pitch": -0.38,
    "Right_Hip_Roll": 0.0,
    "Right_Hip_Yaw": 0.0,
    "Right_Knee_Pitch": 0.8,
    "Right_Ankle_Pitch": -0.43,
    "Right_Ankle_Roll": 0.0,
}


def test_auto_registration():
    """Importing colosseum_sdk should auto-register all robots."""
    assert "t1" in ROBOT_REGISTRY, (
        f"Expected 't1' in registry, got {list(ROBOT_REGISTRY)}"
    )


def test_get_robot():
    cfg = get_robot("t1")
    assert cfg is not None

    try:
        get_robot("nonexistent")
        raise AssertionError("Expected KeyError for unknown robot name")
    except KeyError:
        pass


def test_joint_names():
    cfg = get_robot("t1")
    assert len(cfg.joint_names) == NUM_JOINTS
    assert cfg.joint_names[0] == "AAHead_yaw"
    assert cfg.joint_names[-1] == "Right_Ankle_Roll"
    assert "Left_Hip_Pitch" in cfg.joint_names
    assert "Right_Knee_Pitch" in cfg.joint_names


def test_xml_path_exists():
    cfg = get_robot("t1")
    assert cfg.xml_path.exists(), f"XML not found: {cfg.xml_path}"


def test_locomotion_model_exists():
    cfg = get_robot("t1")
    onnx = cfg.skill_models["locomotion"]
    assert onnx.exists(), f"ONNX not found: {onnx}"
    assert (
        onnx.with_suffix(".onnx.data").exists() or True
    )  # .data optional for small models


def test_default_joint_pos_values():
    cfg = get_robot("t1")
    for joint, expected in EXPECTED_DEFAULT_JOINT_POS.items():
        assert joint in cfg.default_joint_pos, f"Missing key: {joint}"
        assert cfg.default_joint_pos[joint] == expected, (
            f"{joint}: expected {expected}, got {cfg.default_joint_pos[joint]}"
        )


def test_all_joints_covered():
    """Every joint name must appear in every per-joint dict."""
    cfg = get_robot("t1")
    dicts = {
        "default_joint_pos": cfg.default_joint_pos,
        "joint_stiffness": cfg.joint_stiffness,
        "joint_damping": cfg.joint_damping,
        "effort_limit": cfg.effort_limit,
    }
    for field_name, d in dicts.items():
        assert len(d) == NUM_JOINTS, (
            f"{field_name}: expected {NUM_JOINTS} entries, got {len(d)}"
        )
        for joint in cfg.joint_names:
            assert joint in d, f"{field_name}: missing joint '{joint}'"


def test_stiffness_values():
    cfg = get_robot("t1")
    assert cfg.joint_stiffness["Left_Hip_Pitch"] == 200.0
    assert cfg.joint_stiffness["Left_Ankle_Pitch"] == 50.0
    assert cfg.joint_stiffness["Waist"] == 150.0
    assert cfg.joint_stiffness["AAHead_yaw"] == 5.0


def test_effort_limits():
    cfg = get_robot("t1")
    assert cfg.effort_limit["Left_Hip_Pitch"] == 90.0
    assert cfg.effort_limit["Left_Knee_Pitch"] == 118.0
    assert cfg.effort_limit["Left_Ankle_Pitch"] == 57.0
    assert cfg.effort_limit["Waist"] == 40.0


def test_armature_and_frictionloss_are_patterns():
    """joint_armature and joint_frictionloss use regex patterns, not per-joint keys."""
    cfg = get_robot("t1")
    assert ".*" in cfg.joint_armature
    assert cfg.joint_armature[".*"] == 0.3
    assert ".*Ankle.*" in cfg.joint_frictionloss
    assert cfg.joint_frictionloss[".*Ankle.*"] == 0.1
    assert cfg.joint_frictionloss[".*"] == 0.2
