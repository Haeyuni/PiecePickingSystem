from pathlib import Path
import numpy as np


class RobotExecutor:
    """Only validates existing interfaces. It never substitutes direct joint control."""
    def __init__(self, node, config):
        self._node, self._config = node, config

    def preflight(self, execute):
        missing = []
        if not Path(self._config['handeye_path']).is_file():
            missing.append('HANDEYE_MISSING')
        if not self._config.get('motion_action'):
            missing.append('MOTION_ACTION_UNCONFIGURED')
        if not self._config.get('rg2_command_action') or not self._config.get('rg2_state_topic'):
            missing.append('RG2_ROS_INTERFACE_UNCONFIGURED')
        if execute and missing:
            return False, 'DRY_RUN_ONLY:' + ','.join(missing)
        return not missing, ','.join(missing)

    def transform_and_validate(self, candidate):
        # The model result is camera-frame metres. A live TCP transform is mandatory for a robot pose.
        if any(candidate.get(key) is None for key in ('x_m', 'y_m', 'z_m')):
            return None, False, 'CANDIDATE_POSITION_MISSING'
        if not Path(self._config['handeye_path']).is_file():
            return None, False, 'HANDEYE_MISSING'
        # The repository only exposes TCP read, not a verified motion/IK action. Do not invent one.
        return None, False, 'DRY_RUN_ONLY:MOTION_AND_IK_INTERFACE_UNAVAILABLE'

    def execute_pick(self, robot_pose):
        return False, 'DRY_RUN_ONLY:MOTION_AND_RG2_EXECUTION_UNAVAILABLE'
