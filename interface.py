import os
from rich import print
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import gradio as gr

# to get chat response
def get_chat_response(user_input):
    client = MistralClient(api_key=os.environ.get("MISTRAL_API_KEY"))
    chat_response = client.chat(
        model="[ENTER_MISTRAL_FINE_TUNED_MODEL_HERE]",
        messages=[ChatMessage(role='user', content=user_input)]
    )
    content = chat_response.choices[0].message.content
    return content

# gradio interface
iface = gr.Interface(
    fn=get_chat_response,
    inputs="text",
    outputs="text",
    title="Mistral AI Chatbot",
    description="Ask the chatbot any question.",
)

iface.launch()
