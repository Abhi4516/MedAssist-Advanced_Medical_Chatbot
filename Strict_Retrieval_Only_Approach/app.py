

from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import uuid
import yaml
from datetime import datetime

from custom_logging.logger import get_logger
from database.session_db import init_session_db, add_session, get_sessions, clear_old_sessions
from database.chat_history_db import init_db, add_chat_history, get_chat_history, clear_chat_history
from Strict_Retrieval_Only_Approach.rag_pipeline import RAGPipelineManager

app = Flask(__name__)
app.secret_key = "your-secret-key"

logger = get_logger("langchain_integration_log")

with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_dir = config['training']['output_dir']
faiss_index_path = "faiss_index"
default_instruction = config.get("default_instruction", "If you are a doctor, please provide a medically accurate response.")

MAX_SESSIONS = 6


init_session_db()
init_db()


rag_manager = RAGPipelineManager(model_path=model_dir, index_path=faiss_index_path, device="cpu")

def create_new_session():
   
    sid = str(uuid.uuid4())
    creation_time = datetime.now().isoformat()
    add_session(sid, creation_time)
    clear_old_sessions(MAX_SESSIONS)
    return sid

def get_session_id():
    if "session_id" not in session:
        new_id = create_new_session()
        session["session_id"] = new_id
    return session["session_id"]

@app.route("/")
def session_select():
    sessions = get_sessions(MAX_SESSIONS)
    return render_template("session.html", sessions=sessions)

@app.route("/set_session", methods=["POST"])
def set_session():
    chosen_session = request.form.get("session_id")
    if chosen_session:
        session["session_id"] = chosen_session
        logger.info(f"Continuing session {chosen_session}")
        return redirect(url_for("chat_interface"))
    else:
        return redirect(url_for("session_select"))

@app.route("/new_session", methods=["POST"])
def new_session():
    sid = create_new_session()
    session["session_id"] = sid
    logger.info(f"New session created: {sid}")
    return redirect(url_for("chat_interface"))

@app.route("/chat_interface")
def chat_interface():

    if "session_id" not in session:
        return redirect(url_for("session_select"))
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if "session_id" not in session:
        return jsonify({"error": "No session set."}), 400

    sid = session["session_id"]
    data = request.json or {}
  
    user_inp = data.get("input", "").strip()
    
    instruction = data.get("instruction", "").strip() or default_instruction
   
    user_query = f"{instruction}\n{user_inp}"

    # Here we do the "strict retrieval" approach
    try:
        answer = rag_manager.ask_question_strict(user_query)
       
        add_chat_history(sid, instruction, user_inp, answer)
        logger.info(f"Session {sid} => user: {user_query}, answer: {answer[:50]}...")
        return jsonify({"response": answer})
    except Exception as e:
        logger.error(f"Error in /chat for session {sid}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/history", methods=["GET"])
def get_history_route():
    if "session_id" not in session:
        return jsonify({"history": []})
    sid = session["session_id"]
    try:
        rows = get_chat_history(sid, limit=5)
        results = []
        for (t, instr, inp, resp) in rows:
            results.append({
                "timestamp": t,
                "instruction": instr,
                "input": inp,
                "response": resp
            })
        return jsonify({"history": results})
    except Exception as e:
        logger.error(f"Error retrieving history for {sid}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history_route():
    if "session_id" not in session:
        return jsonify({"error": "No session set."}), 400
    sid = session["session_id"]
    try:
        clear_chat_history(sid)
        logger.info(f"Cleared chat for session {sid}")
        return jsonify({"status": "cleared"})
    except Exception as e:
        logger.error(f"Error clearing history for {sid}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
