import spacy
import random

greetings = ["hi", "hello", "hey", "morning", "afternoon", "yo", "hellow", "wazzzaa", "wadup"]
greeting_responses = [
    "Hi!",
    "Hello, how can I help you?",
    "Hi there!",
    "Hey!",
    "WAAAAAAAAAAADUUUUUUUUP",
]
welcome_responses = [
    "Hi there! I'm a bot and you can say hi to me.",
    "Hello! I'm a greeting bot.",
    "Welcome, feel free to say hi to me anytime.",
    "Hey human! I'm a bot, but you can say hi to me and I'll do my best to try and answer.",
]
questions = ["how", "what", "want"]
targets_self = ["bot", "you", "chatbot"]
actions = ["doing", "going"]
objects = ["coffee"]
self_state_responses = [
    "I'm doing fine thank you.",
    "Thanks for asking, I'm doing alright.",
    "Right now I'm feeling great! Just a little sleepy.",
]
action_responses = [
    "Not much really, just hanging",
    "Actually, I have been reading a good book lately, it talks about robots taking over the worl..... not much",
    "BEEP BOOP BAP BEEP BOOOP BOOP BEEP BAP",
]
coffee_responses = [
    "Yes please, that would be great. Please pour it in your ethernet port",
    "No thank you, I'm alergic to caffeine",
]


class AI():


    def __init__(self):
        self.nlp = spacy.load('model')

    def message(self, msg):
        doc = self.nlp(msg)

        label_dict = {t.dep_ : t for t in doc}
        print(f"label_dict: {label_dict}")

        responses = []

       # Revisar si el ROOT es un saludo conocido
        if label_dict["ROOT"].text.lower() in greetings:
            # Responder con un saludo aleatorio
            responses += [random.choice(greeting_responses)]
        elif label_dict["ROOT"].text.lower() in questions:
            # Es una pregunta
            # Responder si preguntan cómo estamos
            if "STATE" in label_dict and label_dict["TARGET"].text.lower() in targets_self:
                responses += [random.choice(self_state_responses)]
            elif "TARGET" in label_dict and label_dict["TARGET"].text.lower() in actions:
                responses += [random.choice(action_responses)]
            elif "OBJECT" in label_dict and label_dict["OBJECT"].text.lower() in objects:
                responses += [random.choice(coffee_responses)]
            else:
                responses += ["I'm sorry, I'm not sure how to answer that."]
            
        else:
            # Responder con un mensaje de bienvenida aleatorio
            responses += [random.choice(welcome_responses)]

        print("Response:")
        print(responses)

        return ' '.join(responses)