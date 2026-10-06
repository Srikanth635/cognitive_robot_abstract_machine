# %% Imports
from cram_viz.onboard.demo import Recorder, derive_segments


# %% Multi-phase action segments
class TestMultiPhaseActionSegments:
    def test_later_pouring_plan_gets_its_own_highlight_segment(self) -> None:
        """
        Ensure a separately compiled pour becomes a timed browser episode.
        """
        recorder = Recorder()
        recorder.frames = [{} for _ in range(30)]
        recorder.base_frames = [[0.0, 0.0, 0.0] for _ in recorder.frames]
        recorder.obj_frames = [
            {
                "Static_MilkPitcher.stl": [
                    min(1.0, max(0.0, (frame_index - 11) / 15)),
                    0.0,
                    0.0,
                ]
            }
            for frame_index in range(len(recorder.frames))
        ]
        recorder.actions = [
            {"action": "ParkArmsAction", "arm": None, "target": None, "frame": 0},
            {"action": "MoveTorsoAction", "arm": None, "target": None, "frame": 0},
            {
                "action": "PickUpAction",
                "arm": "Arms.RIGHT",
                "target": "Static_MilkPitcher.stl",
                "frame": 0,
            },
            {
                "action": "ReachAction",
                "arm": "Arms.RIGHT",
                "target": "Static_MilkPitcher.stl",
                "frame": 0,
            },
            {
                "action": "PouringAction",
                "arm": "Arms.RIGHT",
                "target": "bowl.stl",
                "frame": 22,
            },
        ]

        segments = derive_segments(recorder)

        assert [
            (segment["step"], segment["action"], segment["start"], segment["end"])
            for segment in segments
        ] == [
            ("prepare", "ParkArmsAction,MoveTorsoAction", 0, 12),
            ("pickup_Static_MilkPitcher", "PickUpAction", 12, 22),
            ("pouring_bowl", "PouringAction", 22, 29),
        ]
