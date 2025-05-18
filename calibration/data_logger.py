# calibration/data_logger.py

import csv
import os

LOG_FILE = "calibration/shot_log.csv"

def log_shot(shot_input, cue_ball, object_ball, real_obj=None):
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "angle", "speed", "spin",
                "cue_x", "cue_y",
                "obj_x", "obj_y"
            ])
        row = [
            shot_input.get("angle_deg"),
            shot_input.get("speed"),
            shot_input.get("spin"),
            round(cue_ball.x, 4), round(cue_ball.y, 4),
            round(object_ball.x, 4), round(object_ball.y, 4),
        ]
        if real_obj:
            row.extend([round(real_obj[0], 4), round(real_obj[1], 4)])
        writer.writerow(row)


    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            shot_input.get("angle_deg"),
            shot_input.get("speed"),
            shot_input.get("spin"),
            round(cue_ball.x, 4), round(cue_ball.y, 4),
            round(object_ball.x, 4), round(object_ball.y, 4),
        ])
