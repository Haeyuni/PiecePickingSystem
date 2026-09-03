import time
from onrobot import RG

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

gripper.move_gripper(500)
gripper.close_gripper()

gripper.open_gripper()
gripper.close_gripper()