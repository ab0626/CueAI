from config import *
from physics.simulation import PoolSimulation
from ui.visualizer import PoolVisualizer
from ui.controls import get_user_shot_input
from calibration.data_logger import log_shot
from config import USE_ML_CORRECTION
from calibration.ml_model import predict_correction
from ui.controls import get_real_outcome

def main():
    print ("============= Precision Pool Simulator =============")

    # initialize the simulation + visualizer
    simulation = PoolSimulation(TABLE_WIDTH, TABLE_HEIGHT)
    visualizer = PoolVisualizer(TABLE_WIDTH, TABLE_HEIGHT, POCKET_RADIUS)

    # main loop
    while True:
        # 1. Get user shot input
        shot_input = visualizer.wait_for_shot_input(
            (simulation.cue_ball.x, simulation.cue_ball.y, simulation.cue_ball.radius)
        )

        if shot_input is None:
            print("Exiting the simulation.")
            break

        # 2. Apply shot
        simulation.reset_balls()
        simulation.apply_shot(**shot_input)

        # 3. Run the simulation until balls come to rest
        time_elapsed = 0
        while time_elapsed < MAX_SIM_TIME and not simulation.all_stopped():
            simulation.step(TIME_STEP)
            visualizer.draw(simulation.get_ball_states())
            time_elapsed += TIME_STEP
        
        log_shot(shot_input, simulation.cue_ball, simulation.object_ball)

        print("Shot complete.\n")

        if USE_ML_CORRECTION:
            corrected = predict_correction({
                "angle": shot_input["angle_deg"],
                "speed": shot_input["speed"],
                "spin": shot_input["spin"]
            })

            print(f"\n[ML Prediction] Adjusted object ball target: x={corrected[0]:.3f}, y={corrected[1]:.3f}")
        
        visualizer.play_replay(simulation.get_replay())
        real_result = get_real_outcome()
        log_shot(shot_input, simulation.cue_ball, simulation.object_ball, real_obj=real_result)


        
if __name__ == "__main__":
    main()
