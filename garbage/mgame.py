import pygame
import sys
import cv2
import mediapipe as mp
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Steampunk Elemental Duel")

# Frame rate
FPS = 60
CLOCK = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
FIRE_COLOR = (255, 69, 0)
WATER_COLOR = (0, 191, 255)
ELECTRICITY_COLOR = (255, 255, 0)

# Character properties
CHAR_WIDTH, CHAR_HEIGHT = 50, 80
char1_pos = [100, HEIGHT // 2 - CHAR_HEIGHT // 2]
char2_pos = [WIDTH - 150, HEIGHT // 2 - CHAR_HEIGHT // 2]
CHAR_SPEED = 5
char1_health = 100
char2_health = 100

# Attack properties
ATTACK_SPEED = 10
attacks = []

# Fonts
FONT = pygame.font.SysFont('Arial', 24)

game_over = False

# OpenCV and MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
cap = cv2.VideoCapture(0)

def is_stronger(element1, element2):
    element_strengths = {
        'Fire': 'Electricity',
        'Electricity': 'Water',
        'Water': 'Fire'
    }
    return element_strengths[element1] == element2

def draw_characters():
    # Placeholder rectangles for characters
    pygame.draw.rect(SCREEN, WHITE, (*char1_pos, CHAR_WIDTH, CHAR_HEIGHT))
    pygame.draw.rect(SCREEN, WHITE, (*char2_pos, CHAR_WIDTH, CHAR_HEIGHT))

def draw_health_bars():
    # Player 1 health bar
    pygame.draw.rect(SCREEN, (255,0,0), (50, 20, 200, 20))
    pygame.draw.rect(SCREEN, (0,255,0), (50, 20, 200 * (char1_health / 100), 20))
    # Player 2 health bar
    pygame.draw.rect(SCREEN, (255,0,0), (WIDTH - 250, 20, 200, 20))
    pygame.draw.rect(SCREEN, (0,255,0), (WIDTH - 250, 20, 200 * (char2_health / 100), 20))

def handle_attacks():
    global char1_health, char2_health, game_over
    attacks_to_remove = set()
    # Move and draw attacks
    for i, attack in enumerate(attacks):
        # Move attack
        attack['pos'][0] += attack['speed']
        # Draw attack
        pygame.draw.circle(SCREEN, attack['color'], attack['pos'], 10)
        # Remove attack if off-screen
        if attack['pos'][0] < 0 or attack['pos'][0] > WIDTH:
            attacks_to_remove.add(i)
            continue
    # Check for collisions between attacks
    for i in range(len(attacks)):
        for j in range(i+1, len(attacks)):
            attack1 = attacks[i]
            attack2 = attacks[j]
            # If attacks belong to different owners
            if attack1['owner'] != attack2['owner']:
                # Check if attacks are close enough
                dx = attack1['pos'][0] - attack2['pos'][0]
                dy = attack1['pos'][1] - attack2['pos'][1]
                distance = (dx**2 + dy**2)**0.5
                if distance < 20:  # Both attacks have radius 10
                    # Determine which attack is stronger
                    if attack1['element'] == attack2['element']:
                        # Both attacks are the same element, both are destroyed
                        attacks_to_remove.add(i)
                        attacks_to_remove.add(j)
                    elif is_stronger(attack1['element'], attack2['element']):
                        # attack1 is stronger
                        attacks_to_remove.add(j)
                    elif is_stronger(attack2['element'], attack1['element']):
                        # attack2 is stronger
                        attacks_to_remove.add(i)
                    else:
                        # Neither is stronger, both are destroyed
                        attacks_to_remove.add(i)
                        attacks_to_remove.add(j)
    # Check for collision with opponents
    for i, attack in enumerate(attacks):
        if i in attacks_to_remove:
            continue
        opponent_rect = pygame.Rect(*char2_pos, CHAR_WIDTH, CHAR_HEIGHT) if attack['owner'] == 1 else pygame.Rect(*char1_pos, CHAR_WIDTH, CHAR_HEIGHT)
        if opponent_rect.collidepoint(attack['pos']):
            # Reduce opponent's health
            if attack['owner'] == 1:
                char2_health -= 10  # Adjust damage value as needed
                if char2_health < 0:
                    char2_health = 0
                print(f"Player 1 hit Player 2!")
            else:
                char1_health -= 10
                if char1_health < 0:
                    char1_health = 0
                print(f"Player 2 hit Player 1!")
            attacks_to_remove.add(i)
    # Remove attacks marked for removal
    for index in sorted(attacks_to_remove, reverse=True):
        del attacks[index]
    # Check for game over
    if char1_health <= 0 or char2_health <= 0:
        game_over = True

def count_fingers(hand_landmarks):
    # Returns the number of fingers extended
    finger_tips = [4, 8, 12, 16, 20]
    count = 0
    # Thumb
    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        count += 1
    # Other fingers
    for tip_idx in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP]:
        tip_y = hand_landmarks.landmark[tip_idx].y
        dip_y = hand_landmarks.landmark[tip_idx - 2].y
        if tip_y < dip_y:
            count += 1
    return count

def main():
    global game_over, char1_pos, char2_pos
    running = True
    while running:
        CLOCK.tick(FPS)
        SCREEN.fill(GRAY)

        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # OpenCV frame capture and processing
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Prepare variables for hand data
        if results.multi_hand_landmarks:
            for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label: Left or Right
                hand_label = hand_info.classification[0].label
                # Get hand coordinates in terms of Pygame window
                hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * WIDTH)
                hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * HEIGHT)
                # Count fingers
                fingers = count_fingers(hand_landmarks)

                # Assign hand to player based on label
                if hand_label == 'Left':
                    # Player 1
                    # Update player 1 position
                    char1_pos[1] = hand_y - CHAR_HEIGHT // 2
                    # Ensure character stays on screen
                    char1_pos[1] = max(0, min(HEIGHT - CHAR_HEIGHT, char1_pos[1]))
                    # Attack based on fingers count
                    if fingers == 0:
                        # Fire attack
                        attacks.append({
                            'owner': 1,
                            'pos': [char1_pos[0] + CHAR_WIDTH, char1_pos[1] + CHAR_HEIGHT // 2],
                            'speed': ATTACK_SPEED,
                            'color': FIRE_COLOR,
                            'element': 'Fire'
                        })
                    elif fingers == 1:
                        # Water attack
                        attacks.append({
                            'owner': 1,
                            'pos': [char1_pos[0] + CHAR_WIDTH, char1_pos[1] + CHAR_HEIGHT // 2],
                            'speed': ATTACK_SPEED,
                            'color': WATER_COLOR,
                            'element': 'Water'
                        })
                    elif fingers == 2:
                        # Electricity attack
                        attacks.append({
                            'owner': 1,
                            'pos': [char1_pos[0] + CHAR_WIDTH, char1_pos[1] + CHAR_HEIGHT // 2],
                            'speed': ATTACK_SPEED,
                            'color': ELECTRICITY_COLOR,
                            'element': 'Electricity'
                        })
                elif hand_label == 'Right':
                    # Player 2
                    # Update player 2 position
                    char2_pos[1] = hand_y - CHAR_HEIGHT // 2
                    # Ensure character stays on screen
                    char2_pos[1] = max(0, min(HEIGHT - CHAR_HEIGHT, char2_pos[1]))
                    # Attack based on fingers count
                    if fingers == 0:
                        # Fire attack
                        attacks.append({
                            'owner': 2,
                            'pos': [char2_pos[0], char2_pos[1] + CHAR_HEIGHT // 2],
                            'speed': -ATTACK_SPEED,
                            'color': FIRE_COLOR,
                            'element': 'Fire'
                        })
                    elif fingers == 1:
                        # Water attack
                        attacks.append({
                            'owner': 2,
                            'pos': [char2_pos[0], char2_pos[1] + CHAR_HEIGHT // 2],
                            'speed': -ATTACK_SPEED,
                            'color': WATER_COLOR,
                            'element': 'Water'
                        })
                    elif fingers == 2:
                        # Electricity attack
                        attacks.append({
                            'owner': 2,
                            'pos': [char2_pos[0], char2_pos[1] + CHAR_HEIGHT // 2],
                            'speed': -ATTACK_SPEED,
                            'color': ELECTRICITY_COLOR,
                            'element': 'Electricity'
                        })

                # Draw hand landmarks on the frame (optional)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        draw_characters()
        handle_attacks()
        draw_health_bars()

        if game_over:
            if char1_health <= 0:
                winner_text = "Player 2 Wins!"
            else:
                winner_text = "Player 1 Wins!"
            text = FONT.render(winner_text, True, WHITE)
            text_rect = text.get_rect(center=(WIDTH//2, HEIGHT//2))
            SCREEN.blit(text, text_rect)
            pygame.display.flip()
            pygame.time.wait(3000)
            break  # Exit the game loop

        pygame.display.flip()

        # Display the OpenCV frame (optional)
        cv2.imshow("Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
