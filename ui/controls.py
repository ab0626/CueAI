def get_user_shot_input():
    try:
        angle = float(input("Enter shot angle (degrees, 0 = right): "))
        speed = float(input("Enter shot speed (m/s): "))
        spin = input ("Enter spin (top, back, left, right, none): ").strip().lower()

        return {
            "angle_deg": angle,
            "speed": speed,
            "spin": spin
        }
    except KeyboardInterrupt:
        return None
    except:
        print("Invalid input. Try again.")
        return get_user_shot_input()

def get_real_outcome():
    try:
        real_x = float(input("Enter observed object ball X (m): "))
        real_y = float(input("Enter observed object ball Y (m): "))
        return real_x, real_y
    except:
        return None
