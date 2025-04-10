AgiBotWorld_CONFIG = {
    "images": {
        "head": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_center_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_depth": {
            "dtype": "image",
            "shape": (480, 640, 1),
            "names": ["height", "width", "channel"],
        },
        "head_left_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "head_right_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "hand_left": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "hand_right": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "rgb"],
        },
        "back_left_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
        "back_right_fisheye": {
            "dtype": "video",
            "shape": (748, 960, 3),
            "names": ["height", "width", "rgb"],
        },
    },
    "states": {
        "effector.position": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"motors": ["left_gripper", "right_gripper"]},
        },
        "end.orientation": {"dtype": "float32", "shape": (2, 4), "names": {"motors": ["left_xyzw", "right_xyzw"]}},
        "end.position": {"dtype": "float32", "shape": (2, 3), "names": {"motors": ["left_xyz", "right_xyz"]}},
        "head.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["yaw", "patch"]}},
        "joint.current_value": {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "motors": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                ]
            },
        },
        "joint.position": {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "motors": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                ]
            },
        },
        "robot.orientation": {"dtype": "float32", "shape": (4,), "names": {"motors": ["x", "y", "z", "w"]}},
        "robot.position": {"dtype": "float32", "shape": (3,), "names": {"motors": ["x", "y", "z"]}},
        "waist.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["pitch", "lift"]}},
    },
    "actions": {
        "effector.position": {
            "dtype": "float32",
            "shape": (2,),
            "names": {"motors": ["left_gripper", "right_gripper"]},
        },
        "end.orientation": {"dtype": "float32", "shape": (2, 4), "names": {"motors": ["left_xyzw", "right_xyzw"]}},
        "end.position": {"dtype": "float32", "shape": (2, 3), "names": {"motors": ["left_xyz", "right_xyz"]}},
        "head.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["yaw", "patch"]}},
        "joint.position": {
            "dtype": "float32",
            "shape": (14,),
            "names": {
                "motors": [
                    "left_arm_0",
                    "left_arm_1",
                    "left_arm_2",
                    "left_arm_3",
                    "left_arm_4",
                    "left_arm_5",
                    "left_arm_6",
                    "right_arm_0",
                    "right_arm_1",
                    "right_arm_2",
                    "right_arm_3",
                    "right_arm_4",
                    "right_arm_5",
                    "right_arm_6",
                ]
            },
        },
        "robot.velocity": {"dtype": "float32", "shape": (2,), "names": {"motors": ["x_vel", "yaw_vel"]}},
        "waist.position": {"dtype": "float32", "shape": (2,), "names": {"motors": ["pitch", "lift"]}},
    },
}
