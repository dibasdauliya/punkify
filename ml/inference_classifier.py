import pickle
import cv2
import mediapipe as mp
import numpy as np
import random
import time

import openai
import speech_recognition as sr
import pyttsx3

# Initialize OpenAI client
client = openai.OpenAI(api_key='sk-')

# Initialize speech recognition
recognizer = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# This list will store the conversation history for memory during the session
conversation_history = []
max_messages_before_summary = 5  # Number of messages before summarizing older ones


def listen_for_speech():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError:
        print("Sorry, there was an error with the speech recognition service.")
        return None


def speak_response(text):
    print("AI response:", text)
    engine.say(text)
    engine.runAndWait()


def summarize_memory():
    """Summarizes the conversation history and retains recent interactions."""
    global conversation_history
    
    if len(conversation_history) > max_messages_before_summary:
        # Only summarize if we have more than the allowed number of exchanges
        recent_conversations = conversation_history[-max_messages_before_summary:]  # Keep recent messages intact
        old_conversations = conversation_history[:-max_messages_before_summary]  # Select older messages for summarization
        
        # Create a summary using GPT
        response = client.chat.completions.create(
            model="gpt-4o",  # Replace with the actual model name
            messages=[
                {"role": "system", "content": "Summarize the following conversation briefly."},
                {"role": "user", "content": "\n".join([f"{msg['role']}: {msg['content']}" for msg in old_conversations])}
     
            ],
            max_tokens=100
        )
        
        summary = response.choices[0].message.content
        print("Conversation summary:", summary)

        # Keep the summary and recent conversation
        conversation_history = [{"role": "system", "content": summary}] + recent_conversations


def chat_with_gpt4o():
    global conversation_history  # Use the session-based memory
    speak_response("Hey friend, do you wanna know about steam punk?")
    while True:
        speak_response("I am listening,")
        user_input = listen_for_speech()

        if user_input:
            if user_input.lower() == "exit":
                print("Exiting the program...")
                return
            
            # Append the user input to the conversation history
            
            conversation_history.append({"role": "assistant", "content": "Hey friend, do you wanna know about steam punk?"})
            conversation_history.append({"role": "user", "content": user_input})

            # Summarize memory if needed
            summarize_memory()

            # Use the full conversation history in the request
            response = client.chat.completions.create(
                model="gpt-4o",  # Replace with the actual model name when available
                messages=[{"role": "system", "content": "Answer like a regular human conversation which means one to two sentence only."}, *conversation_history],
                max_tokens=150
            )
            
            ai_response = response.choices[0].message.content
            
            # Append the AI response to the conversation history
            conversation_history.append({"role": "assistant", "content": ai_response})
            
            speak_response(ai_response)


# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Setup camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands and Face Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_detection = mp.solutions.face_detection

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.3)

# Load the hat images
hats = [
    cv2.imread('hat.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('hat2.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('hat3.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('hat4.png', cv2.IMREAD_UNCHANGED)
]  # Make sure it's transparent PNG for best effect

# Labels dictionary and description dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G'}
description_dict = {
    '0': ('./Pic/pic1.webp', 'This steampunk scene features a vintage-inspired camera and car, beautifully crafted with intricate brass and copper elements and exposed gears that highlight the essence of Victorian industrial style. The camera boasts a retro aesthetic, with leather detailing, clockwork mechanisms, and antique lenses that evoke a sense of timeless craftsmanship. The car, a fusion of early automobile and carriage designs, is adorned with copper pipes, spoked wheels, and polished brass fixtures, exuding both elegance and ruggedness. Together, these pieces capture the imaginative blend of classical engineering and retro-futuristic charm that defines the steampunk genre.'),
    '1': ('./Pic/pic2.webp', 'This steampunk-inspired mechanical airship, adorned with Victorian-industrial details, embodies the spirit of adventure. Crafted from brass and copper, the airship is powered by intricate steam-driven mechanisms, with exposed pipes and gears adding to its grandeur. It floats above a misty cityscape filled with vintage architecture, evoking an atmosphere of mystery and exploration. The foggy surroundings and towering buildings create a dramatic backdrop, enhancing the airship\'s role as both a marvel of engineering and a vessel for limitless journeys in a steampunk world.'),
    '2': ('./Pic/pic3.webp', 'This steampunk-inspired locomotive, adorned with brass, copper, and iron, epitomizes the bold mechanics of Victorian-era transport. Billowing steam surrounds the train as it travels through an industrial landscape, showcasing exposed gears, polished metal pipes, and intricate detailing. Its robust design and powerful steam engine evoke a sense of unstoppable momentum, blending elegance with rugged industrial charm. This locomotive stands as a testament to the artistry and innovation of steampunk, a symbol of progress and adventure in a fantastical, mechanical age.'),
    '3': ('./Pic/pic4.webp', 'This steampunk-inspired submarine, meticulously crafted from brass and copper, glides through a shadowy, sepia-toned underwater world. Riveted panels and round portholes give it a rugged, vintage look, while sturdy propellers drive it forward. Small smokestacks release streams of bubbles, and its glowing headlights cut through the dim depths, casting light across the ocean floor. The submarine embodies the intrigue and craftsmanship of steampunk design, combining adventure with the allure of hidden, unexplored waters.'),
    '4': ('./Pic/pic5.webp', 'This steampunk-inspired wristwatch, with its exposed gears, Roman numerals, and rich brass and copper detailing, is a masterpiece of intricate design. Tiny steam pipes adorn the watch, releasing subtle puffs of mist that add a touch of mystery. Resting on a Victorian-style table, the watch is surrounded by various tools and fittings, immersing it in an industrial, sepia-toned ambiance. This timepiece blends precise mechanics with timeless elegance, capturing the heart of steampunk\’s fusion of artistry and innovation.'),
    '5': ('./Pic/pic6.webp', 'This steampunk-inspired telescope, adorned with intricate brass and copper detailing, evokes the elegance and curiosity of a bygone era. Dark leather accents and exposed gears add depth to its design, while the telescope is positioned within a Victorian observatory overlooking a fog-shrouded industrial cityscape. The warm, sepia-toned lighting bathes the scene, highlighting every polished curve and gear. This telescope is both a tool for discovery and a symbol of steampunk\’s blend of scientific wonder and mechanical artistry, capturing the essence of exploration in a richly atmospheric setting.'),
    '6': ('./Pic/pic7.webp', 'This steampunk-inspired typewriter, with its brass and copper detailing, exposed gears, and vintage round keys, is a tribute to both craftsmanship and style. Nestled in a cozy Victorian study, the typewriter\’s intricate components glint under warm sepia lighting, highlighting each polished surface and delicate gear. The ambiance is rich with industrial charm, yet softened by classical elegance, making the typewriter not just a tool for writing, but a work of art that embodies the beauty of steampunk\’s retro-futuristic aesthetic.'),
    '7': ('./Pic/pic8.webp', 'This steampunk-inspired mechanical clock, showcasing brass and copper detailing with exposed gears and bold Roman numerals, is a stunning fusion of function and artistry. Set within a Victorian industrial room, the clock is framed in dark wood and iron, giving it a solid, timeless feel. Pipes and small steam vents surround the face, releasing gentle puffs of steam that enhance its unique charm. This clock stands as a centerpiece of steampunk design, merging elegance with rugged industrial elements to capture the essence of a bygone era reimagined with mechanical wonder.'),
    '8': ('./Pic/pic9.webp', 'This steampunk-inspired globe, crafted from rich brass and dark wood, brings a sense of exploration and craftsmanship to life. Exposed gears and metal pipes intertwine around the globe, which rests on an ornate stand adorned with intricate brass detailing. Situated in a Victorian study filled with vintage books and scientific instruments, the globe is bathed in warm, sepia-toned light, adding depth and atmosphere. It embodies the spirit of discovery and the artistry of steampunk, merging elegance with the rugged beauty of industrial design in a setting that invites curiosity.'),
    '9': ('./Pic/pic10.webp', 'This steampunk-inspired Victorian ray gun is a marvel of intricate design and retro-futuristic imagination. Crafted with brass, copper, and leather, the gun showcases exposed gears, metal coils, and a vibrant, glowing energy core at its heart. Positioned within a Victorian industrial workshop filled with tools and mechanical parts, this ray gun radiates a sense of power and sophistication. Its blend of detailed craftsmanship with a touch of futuristic technology captures the essence of steampunk, merging elegance with innovation in a weapon that is as visually striking as it is inventive.'),
}

# Variable to store the last predicted character
last_predicted_character = None

# Initialize timer for hat change
last_hat_change_time = time.time()
current_hat = hats[1]
last_description_update_time = time.time()
current_description = description_dict['0']

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Update the last predicted character
        last_predicted_character = predicted_character
        
        if predicted_character == 'E':
            chat_with_gpt4o()

    # Detect faces and add hat if the last predicted character is "A"
    if last_predicted_character == 'A':
        face_results = face_detection.process(frame_rgb)
        if face_results.detections:
            # Check if 20 seconds have passed since the last hat change
            if time.time() - last_hat_change_time > 5:
                current_hat = random.choice(hats)
                last_hat_change_time = time.time()  # Reset the timer

            for detection in face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x, y, w, h = int(bboxC.xmin * W), int(bboxC.ymin * H), int(bboxC.width * W), int(bboxC.height * H)

                # Resize hat image to be three times the width of the detected face
                hat_resized = cv2.resize(current_hat, (w * 3, int(h * 1.5)))
                
                # Calculate position to overlay hat
                y_offset = y - int(h * 1.5)
                y1, y2 = max(0, y_offset), min(frame.shape[0], y_offset + hat_resized.shape[0])
                x1, x2 = max(0, x - w), min(frame.shape[1], x + w * 2)

                # Adjust the resized hat dimensions if they don't fit in the frame
                hat_resized = hat_resized[0:y2 - y1, 0:x2 - x1]

                # Overlay hat image on frame
                if hat_resized.shape[2] == 4:
                    alpha_hat = hat_resized[:, :, 3] / 255.0
                    for c in range(3):
                        frame[y1:y2, x1:x2, c] = (
                            alpha_hat * hat_resized[:, :, c] + (1 - alpha_hat) * frame[y1:y2, x1:x2, c]
                        )

    # Display the image and description if the last predicted character is "B"
    if time.time() - last_description_update_time > 5:
        current_key = random.choice(list(description_dict.keys()))
        current_description = description_dict[current_key]
        last_description_update_time = time.time()

    # Display the selected image and description if last_predicted_character is "B"
    if last_predicted_character == 'B':
        image_path, description = current_description
        image = cv2.imread(image_path)

        if image is not None:
            # Define the box dimensions for half screen height and slightly increased width
            box_width = int(0.5 * W)
            box_height = int(H)  # Half of the screen height
            box = np.ones((box_height, box_width, 3), dtype=np.uint8) * 240  # Light gray background

            # Resize image to fit within the box height and slightly smaller than width
            image_resized = cv2.resize(image, (box_width - 20, int(box_height * 0.7)))

            # Position the image within the box with padding
            padding = 10
            box[padding:padding + image_resized.shape[0], padding:padding + image_resized.shape[1]] = image_resized

            # Set up text position and wrapping width
            text_y_start = image_resized.shape[0] + 30
            max_text_width = box_width - 20

            # Wrap the text to fit within the box width
            lines = []
            current_line = ""
            for word in description.split():
                if cv2.getTextSize(current_line + word, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0] < max_text_width:
                    current_line += word + " "
                else:
                    lines.append(current_line.strip())
                    current_line = word + " "
            lines.append(current_line.strip())  # Add the final line

            # Draw each line of text onto the box
            for i, line in enumerate(lines):
                cv2.putText(
                    box, line, (padding, text_y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
                )

            # Place the box with the image and text onto the frame
            frame[0:box_height, W - box_width:W] = box
            
            
    # Clear the box if the last predicted character is "G"
    elif last_predicted_character == 'G':
        ret, frame = cap.read()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


