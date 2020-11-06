import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window

from messages import Messages
from inputs import Inputs

from ai import AI

from kivy.config import Config
Config.set('graphics', 'width', '400')
Config.set('graphics', 'height', '800')

from kivy.properties import ObjectProperty

class MainScreen(BoxLayout):
    messages=ObjectProperty(None)
    inputs=ObjectProperty(None)

    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        #self.messages
        self.messages.bind(minimum_height=self.messages.setter('height'))

        self.ai = AI()

        self.inputs.set_messages_handler(self.messages)
        self.inputs.set_ai(self.ai)

class ChatbotApp(App):
    def build(self):
        self.title = 'CHATBOT POWERED BY GODBOT'
        return MainScreen()
        # Main widget - MainScreen
        # self.layout = BoxLayout(orientation='vertical', spacing=10)


        # # self.messages = Messages(size_hint=(1, .8))
        # self.messages = Messages(size_hint_y=None)
        # # self.messages.bind(minimum_height=self.messages.setter('height'))

        # # self.hello2 = Label(text='Área de input + botón', size_hint=(1, .2))
        # self.inputs = Inputs(size_hint=(1, .2))

        # self.ai = AI()

        # self.inputs.set_messages_handler(self.messages)
        # self.inputs.set_ai(self.ai)

        # messages_container = ScrollView(size_hint=(.5, None), size=(Window.width, Window.height))

        # # self.layout.add_widget(self.messages)
        # messages_container.add_widget(self.messages)
        # self.layout.add_widget(messages_container)

        # self.layout.add_widget(self.inputs)
        # return self.layout


if __name__ == "__main__":

    ChatbotApp().run()
    