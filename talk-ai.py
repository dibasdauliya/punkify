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
    
    while True:
        speak_response("I am listening,")
        user_input = listen_for_speech()

        if user_input:
            if user_input.lower() == "exit":
                print("Exiting the program...")
                break
            
            # Append the user input to the conversation history
            conversation_history.append({"role": "user", "content": user_input})

            # Summarize memory if needed
            summarize_memory()

            # Use the full conversation history in the request
            response = client.chat.completions.create(
                model="gpt-4o",  # Replace with the actual model name when available
                messages=[{"role": "system", "content": "You are a helpful assistant."}, *conversation_history],
                max_tokens=150
            )
            
            ai_response = response.choices[0].message.content
            
            # Append the AI response to the conversation history
            conversation_history.append({"role": "assistant", "content": ai_response})
            
            speak_response(ai_response)


if __name__ == "__main__":
    speak_response("Start speaking to chat with GPT-4o. Say 'exit' to end the conversation.")
    chat_with_gpt4o()
