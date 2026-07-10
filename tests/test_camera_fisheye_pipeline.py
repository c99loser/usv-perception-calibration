import tempfile
import unittest
from pathlib import Path

import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "camera_fisheye"))

from camera_fisheye_pipeline import (  # noqa: E402
    CAMERA_INFO_TOPIC_TEMPLATE,
    IMAGE_TOPIC_TEMPLATE,
    BagSummary,
    RectificationPlan,
    TopicInfo,
    compute_rectification_map_metrics,
    compute_rectified_image_metrics,
    estimate_intrinsics_from_fov,
    parse_metadata,
    select_camera_topics,
    to_simple_yaml,
    corrected_bag_output_dir,
    write_stage_report,
)


class CameraFisheyePipelineTests(unittest.TestCase):
    def test_select_camera_topics_prefers_direct_topics(self):
        topics = []
        for camera in ("front", "back", "left", "left2", "right", "right2"):
            topics.append(
                TopicInfo(
                    name=IMAGE_TOPIC_TEMPLATE.format(name=camera),
                    msg_type="sensor_msgs/msg/Image",
                    message_count=12,
                )
            )
            topics.append(
                TopicInfo(
                    name=CAMERA_INFO_TOPIC_TEMPLATE.format(name=camera),
                    msg_type="sensor_msgs/msg/CameraInfo",
                    message_count=12,
                )
            )
        summary = BagSummary(
            bag_dir=Path("bag"),
            mcap_file=Path("bag") / "bag.mcap",
            duration_seconds=2.4,
            starting_time_ns=1,
            total_messages=144,
            topics=tuple(topics),
        )

        selections = select_camera_topics(summary)

        self.assertEqual(len(selections), 6)
        self.assertTrue(all(selection.image_count == 12 for selection in selections))
        self.assertEqual(selections[0].image_source, "direct")
        self.assertEqual(selections[0].camera_info_source, "direct")

    def test_parse_metadata_extracts_camera_topic_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = Path(tmp) / "metadata.yaml"
            metadata.write_text(
                """
rosbag2_bagfile_information:
  duration:
    nanoseconds: 2000000000
  starting_time:
    nanoseconds_since_epoch: 1782954197498782013
  message_count: 2
  topics_with_message_count:
    - topic_metadata:
        name: /wamv/sensors/cameras/front_camera_sensor/image_raw
        type: sensor_msgs/msg/Image
      message_count: 8
    - topic_metadata:
        name: /wamv/sensors/cameras/front_camera_sensor/camera_info
        type: sensor_msgs/msg/CameraInfo
      message_count: 8
  files:
    - path: sample_0.mcap
""",
                encoding="utf-8",
            )

            summary = parse_metadata(metadata)

        self.assertEqual(summary.duration_seconds, 2.0)
        self.assertEqual(summary.total_messages, 2)
        self.assertEqual(summary.topics[0].message_count, 8)
        self.assertEqual(summary.mcap_file.name, "sample_0.mcap")

    def test_estimated_intrinsics_are_centered_and_marked_as_fisheye(self):
        intrinsics = estimate_intrinsics_from_fov(
            "front",
            "/wamv/sensors/cameras/front_camera_sensor/image_raw",
        )

        self.assertEqual(intrinsics.width, 1280)
        self.assertEqual(intrinsics.height, 720)
        self.assertAlmostEqual(intrinsics.k[2], 640.0)
        self.assertAlmostEqual(intrinsics.k[5], 360.0)
        self.assertEqual(intrinsics.distortion_model, "fisheye")

    def test_simple_yaml_emits_nested_camera_data(self):
        rendered = to_simple_yaml(
            {
                "cameras": {
                    "front": {
                        "width": 1280,
                        "topic": "/wamv/sensors/cameras/front_camera_sensor/image_raw",
                    }
                }
            }
        )

        self.assertIn("cameras:", rendered)
        self.assertIn("front:", rendered)
        self.assertIn("width: 1280", rendered)

    def test_stage_report_marks_missing_inputs(self):
        summary = BagSummary(
            bag_dir=Path("bag"),
            mcap_file=Path("bag") / "bag.mcap",
            duration_seconds=1.0,
            starting_time_ns=1,
            total_messages=1,
            topics=(),
        )
        selections = select_camera_topics(summary)
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "camera_stage_report.md"
            write_stage_report(report_path, [summary], {summary.bag_dir: selections})
            report = report_path.read_text(encoding="utf-8")

        self.assertIn("missing-input", report)
        self.assertIn("front", report)

    def test_corrected_bag_output_dir_is_under_corrected_mcaps(self):
        output = corrected_bag_output_dir(Path("data/session_a"), Path("output/camera_fisheye"))

        self.assertEqual(
            output,
            Path("output/camera_fisheye/corrected_mcaps/session_a_fisheye_corrected"),
        )

    def test_rectification_map_metrics_count_invalid_pixels(self):
        intrinsics = estimate_intrinsics_from_fov(
            "front",
            "/wamv/sensors/cameras/front_camera_sensor/image_raw",
        )
        intrinsics.width = 3
        intrinsics.height = 2
        map1 = np.asarray(
            [
                [[0, 0], [1, 0], [2, 0]],
                [[0, 1], [1, 1], [-1, 1]],
            ],
            dtype=np.int16,
        )
        map2 = np.zeros((2, 3), dtype=np.uint16)
        plan = RectificationPlan(intrinsics, map1, map2, np.eye(3), "test")

        metrics = compute_rectification_map_metrics(plan)

        self.assertAlmostEqual(metrics["valid_pixel_ratio"], 5.0 / 6.0)
        self.assertAlmostEqual(metrics["invalid_pixel_ratio"], 1.0 / 6.0)
        self.assertAlmostEqual(metrics["roi_area_ratio"], 1.0)

    def test_rectified_image_metrics_count_black_pixels(self):
        image = np.asarray(
            [
                [[0, 0, 0], [8, 0, 0]],
                [[5, 5, 5], [10, 10, 10]],
            ],
            dtype=np.uint8,
        )

        metrics = compute_rectified_image_metrics(image, black_threshold=5)

        self.assertAlmostEqual(metrics["black_pixel_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
