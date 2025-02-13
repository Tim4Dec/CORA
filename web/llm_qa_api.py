import streamlit as st
import  sqlite3
from datetime import datetime, timedelta
from openai import OpenAI

DB_FILE = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cur_dialogs (
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model_name TEXT NOT NULL,
            time TEXT NOT NULL,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')

    cursor.execute('''
           CREATE TABLE IF NOT EXISTS dialog (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               username TEXT NOT NULL,
               model TEXT NOT NULL,
               start_time TEXT NOT NULL,
               FOREIGN KEY (username) REFERENCES users (username)
           )
       ''')

    cursor.execute('''
           CREATE TABLE IF NOT EXISTS utterance (
               dialog_id INTEGER NOT NULL,
               idx INTEGER NOT NULL,
               content TEXT NOT NULL,
               role TEXT NOT NULL,
               time TEXT NOT NULL,
               FOREIGN KEY (dialog_id) REFERENCES dialog (id)
           )
       ''')

    conn.commit()
    conn.close()

def is_new_dialog(username):
    if st.session_state.get('loaded_previous', False):
        st.session_state['loaded_previous'] = False
        return False

    if st.session_state.get('cleared_previous', False):  
        st.session_state['cleared_previous'] = False 
        return True

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT time FROM cur_dialogs WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()

    if row:
        last_time = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        current_time = datetime.now()
        if current_time - last_time > timedelta(minutes=5):
            return True 
        else:
            return False
    else:
        return True 


def save_chat_history(username, model_name, messages):
    if is_new_dialog(username):
        archive_current_dialog(username)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("DELETE FROM cur_dialogs WHERE username=? AND model_name=?", (username, model_name))

    for message in messages:
        cursor.execute(
            "INSERT INTO cur_dialogs (username, role, content, model_name,time) VALUES (?, ?, ?, ?,?)",
            (username, message['role'], message['content'], model_name, current_time)
        )

    conn.commit()
    conn.close()

def archive_current_dialog(username):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT model_name, content, role, time FROM cur_dialogs WHERE username=?", (username,))
    rows = cursor.fetchall()

    if rows:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT INTO dialog (username, model, start_time)
            VALUES (?, ?, ?)
        ''', (username, rows[0][0], start_time))
        dialog_id = cursor.lastrowid 

        for idx, row in enumerate(rows, start=1):
            cursor.execute('''
                INSERT INTO utterance (dialog_id, idx, content, role, time)
                VALUES (?, ?, ?, ?, ?)
            ''', (dialog_id, idx, row[1], row[2], row[3]))

        cursor.execute("DELETE FROM cur_dialogs WHERE username=?", (username,))

    conn.commit()
    conn.close()

def load_chat_history(username, model_name):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM dialog WHERE username=? AND model=? ORDER BY start_time DESC LIMIT 1",
                   (username, model_name))
    row = cursor.fetchone()

    if row:
        dialog_id = row[0]
        st.session_state['dialog_id'] = dialog_id 

        cursor.execute("DELETE FROM utterance WHERE dialog_id=?", (dialog_id,))
        cursor.execute("DELETE FROM dialog WHERE id=? AND username=?", (dialog_id, username))
        conn.commit()
    else:
        st.session_state['dialog_id'] = None  

    cursor.execute("SELECT role, content FROM cur_dialogs WHERE username=? AND model_name=?", (username, model_name))
    rows = cursor.fetchall()
    conn.close()

    st.session_state['loaded_previous'] = True

    return [{'role': row[0], 'content': row[1]} for row in rows]

def authenticate(username, password):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    if row and row[0] == password:
        return True
    return False

def is_username_taken(username):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cursor.fetchone()
    conn.close()
    return row is not None

def register_user(username, password):
    if is_username_taken(username):
        return False  
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()
    return True 

def login():
    st.sidebar.title("User Login")
    username = st.sidebar.text_input("User Name")
    password = st.sidebar.text_input("Password", type="password")

    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if st.sidebar.button("login") and not st.session_state["logged_in"]:
        if authenticate(username, password):
            st.session_state['username'] = username  
            st.session_state['logged_in'] = True
            st.sidebar.success("Login sucess")
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password")

    if st.session_state["logged_in"]:
        chat_interface()  
    else:
        if st.sidebar.button("register"):
            if username and password:
                if register_user(username, password):
                    st.sidebar.success("Register sucess")
                else:
                    st.sidebar.error("Username already taken")
            else:
                st.sidebar.error("Username and password cannot be empty")

api_urls = {
    "ChatGLM-NP": "http://****:5902/v1",  
    "ChatGLM-CP": "http://****:5902/v1",  
    "ChatGLM-TP": "http://****:5902/v1",  
    "Model 4":"http://****:5902/v1", 
    "ChatGLM-FT":"http://****:5902/v1",
}


fs = open('chat_log.txt', 'a+')
fs.write(__file__)


def on_btn_click(model_name):
    st.session_state['cleared_previous'] = True
    save_chat_history(st.session_state['username'], model_name, st.session_state.messages) 
    st.session_state.messages = [] 



def set_config():
    
    if "current_model" not in st.session_state:
        st.session_state["current_model"] = "" 
    
    base_config = {"model_name": ""}
    model_config = {'top_k': '', 'top_p': '', 'temperature': '', 'max_length': ''}

    with st.sidebar:
        
        model_name = st.radio(
            "Model Selection:",
            # ["ChatGLM-NP", "ChatGLM-CP", "ChatGLM-TP", "Model 4","ChatGLM-FT"],
            ["Model 1", "Model 2", "Model 3", "Model 4", "Model 5"],
            index=0
        )
        base_config['model_name'] = model_name

        max_length = st.slider("Max Length", min_value=64, max_value=2048, value=512)
        top_p = st.slider(
            'Top P', 0.0, 1.0, 0.7, step=0.01
        )
        temperature = st.slider(
            'Temperature', 0.0, 2.0, 0.95, step=0.01
        )

    
    if st.session_state["current_model"] != model_name:
        
        if st.session_state["current_model"]:
            save_chat_history(st.session_state['username'], st.session_state["current_model"],
                              st.session_state.messages)
            st.session_state.messages = [] 


    model_config['top_p'] = top_p
    model_config['max_length'] = max_length
    model_config['temperature'] = temperature
    return base_config, model_config

def set_input_format(model_name):
    if model_name == "ChatGLM-NP":
       
        default_prompt = """You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown."""

        
        modified_prompt = st.sidebar.text_area("Edit the system prompt", value=default_prompt, height=120)

        
        input_format = modified_prompt

    elif model_name == "ChatGLM-CP":
        
        default_prompt = """I want you to take on the role of a psychologist. I will provide you with someone seeking guidance and advice to manage their emotions, stress, anxiety, and other mental health issues. You should use your knowledge of cognitive-behavioral therapy to formulate strategies that individuals can implement to improve their overall health."""

        
        modified_prompt = st.sidebar.text_area("System prompt", value=default_prompt, height=120, disabled=True)

        input_format = modified_prompt

    elif model_name == "ChatGLM-TP":
        
        default_prompt = """You are a counselor. Please use cognitive behavioral therapy to improve a client's mental health by asking questions to discover their situation, thoughts, and emotional reactions."""

        modified_prompt = st.sidebar.text_area("System prompt", value=default_prompt, height=120, disabled=True)

        input_format = modified_prompt

    elif model_name == "Model 4":
        default_prompt = """ """ 
        input_format = st.sidebar.text_area("System prompt", value=default_prompt, height=0, disabled=True)

    elif model_name == "ChatGLM-FT":
        default_prompt = """ """
        input_format = st.sidebar.text_area("System prompt", value=default_prompt, height=0, disabled=True)

    else:
        input_format = ""

    return input_format


def llm_chat_via_api(model_name, input_format, messages, model_config):
    
    api_url = api_urls.get(model_name, "http://****:5902/v1") 

    
    system_message = {"role": "system", "content": input_format}


    # context_messages = messages[-5:] if len(messages) > 4 else messages
    context_messages = messages[-11:] if len(messages) > 10 else messages
    # context_messages = messages


    full_messages = [system_message] + context_messages

    client = OpenAI(api_key="0", base_url=api_url)

    result = client.chat.completions.create(
        model=model_name,
        messages=full_messages,  
        top_p=model_config['top_p'],
        max_tokens=model_config['max_length'],
        temperature=model_config['temperature']
    )

    if result.choices:
        return result.choices[0].message.content
    else:
        return "API inference failed or no result"


def chat_interface():
    st.title(f"Welcome, {st.session_state['username']}!")

    if 'messages' not in st.session_state:
        st.session_state.messages = []  

    base_config, model_config = set_config()
    model_name = base_config['model_name']
    input_format = set_input_format(model_name=model_name)


    if st.session_state["current_model"] != model_name:
        st.session_state["current_model"] = model_name
        if model_name in ["ChatGLM-FT", "Model 4"]:
            start_message = "Start Conversation>>>"
        else:
            start_message = "Hello"

        st.session_state.messages.append({"role": "user", "content": start_message, "show": False})

        cur_response = llm_chat_via_api(model_name, input_format, st.session_state.messages, model_config)

        st.session_state.messages.append({"role": "assistant", "content": cur_response, "avatar": "🤖", "show": True})

    if st.sidebar.button("Load Chat History", key="load_chat_history"):
        st.session_state.messages = load_chat_history(st.session_state['username'], model_name)

    st.sidebar.button("Clear Chat History", on_click=lambda: on_btn_click(model_name), key="clear_chat_history")

    st.header(f'Large Language Model：{model_name}')


    for message in st.session_state.messages:
        if message.get("show", True):  
            with st.chat_message(message["role"], avatar=message.get("avatar")):
                st.markdown(message["content"])

    if user_query := st.chat_input("Please enter content..."):
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query, "avatar": "🧑‍💻", "show": True})


        with st.chat_message("assistant", avatar="🤖"):
            max_len = model_config['max_length']
            if len(user_query) > max_len:
                cur_response = f'Word count exceeds {max_len}, please try again.'
            else:
                cur_response = llm_chat_via_api(model_name, input_format, st.session_state.messages, model_config)

            st.markdown(cur_response)
            st.session_state.messages.append({"role": "assistant", "content": cur_response, "avatar": "🤖", "show": True})

        save_chat_history(st.session_state['username'], model_name, st.session_state.messages)

if __name__ == "__main__":
    init_db()  
    if 'username' not in st.session_state:
        login()  
    else:
        chat_interface() 