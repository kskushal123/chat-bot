import os
from flask import Flask, render_template, request, jsonify
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage, AssistantMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

ENDPOINT = "https://models.github.ai/inference"
MODEL = "deepseek/DeepSeek-V3-0324"
TOKEN = os.environ.get("GITHUB_TOKEN", "")

client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(TOKEN),
)

# In-memory conversation history per session (simple approach)
conversation_history = []

SYSTEM_PROMPT = (
    "You are a helpful, friendly, and knowledgeable AI assistant. "
    "Respond clearly and concisely. Use markdown formatting where appropriate."
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    # Build messages list
    messages = [SystemMessage(SYSTEM_PROMPT)]
    for turn in conversation_history:
        if turn["role"] == "user":
            messages.append(UserMessage(turn["content"]))
        else:
            messages.append(AssistantMessage(content=turn["content"]))
    messages.append(UserMessage(user_message))

    try:
        response = client.complete(
            messages=messages,
            model=MODEL,
        )
        assistant_reply = response.choices[0].message.content

        # Save to history
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "assistant", "content": assistant_reply})

        return jsonify({"reply": assistant_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clear", methods=["POST"])
def clear():
    conversation_history.clear()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
