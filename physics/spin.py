# physics/spin.py

import math

def apply_spin_force(ball, spin_type):
    # crude approximation of side force from spin
    if spin_type == 'left':
        ball.vx += 0.1 * ball.vy
        ball.vy -= 0.1 * ball.vx
    elif spin_type == 'right':
        ball.vx -= 0.1 * ball.vy
        ball.vy += 0.1 * ball.vx
    elif spin_type == 'top':
        ball.vx *= 1.02
        ball.vy *= 1.02
    elif spin_type == 'back':
        ball.vx *= 0.98
        ball.vy *= 0.98
