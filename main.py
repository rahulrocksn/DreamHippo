import sys
import subprocess
import importlib

# Auto-install required packages
def install_dependencies():
    """Checks and installs required packages automatically."""
    required_packages = ["openai", "flask"]
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"📦 Installing missing dependency: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed!")

# Run installation before other imports
install_dependencies()

import os
import openai

import json
import time
import math
import random
from typing import Dict, List, Optional, Any
from openai import OpenAI
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# ==================================================================================
# 1. CORE INFRASTRUCTURE
# ==================================================================================

def get_api_key():
    """Retrieves the OpenAI API key securely from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY environment variable not found.")
        print("Please set it in your terminal: export OPENAI_API_KEY='sk-...'")
        return None
    return api_key

import random

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x + random.uniform(0, 1))
                    print(f"\033[93m[Network] Error {e}, retrying in {sleep:.2f}s...\033[0m")
                    time.sleep(sleep)
                    x += 1
        return wrapper
    return decorator

# Global Token Tracker
TOKEN_USAGE = {"input": 0, "output": 0}

@retry_with_backoff(retries=3)
def call_model(
    messages: List[Dict[str, str]], 
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False
) -> str:
    """
    Wrapper for OpenAI API calls.
    """
    api_key = get_api_key()
    if not api_key:
        return ""
        
    client = OpenAI(api_key=api_key)
    
    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    
    resp = client.chat.completions.create(**kwargs)
    
    # Track Usage
    if resp.usage:
        TOKEN_USAGE["input"] += resp.usage.prompt_tokens
        TOKEN_USAGE["output"] += resp.usage.completion_tokens
        
    return resp.choices[0].message.content or ""

def parse_json_output(response_text: str) -> Dict[str, Any]:
    """Helper to parse JSON from LLM output, handling code blocks if present."""
    clean_text = response_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text.replace("```json", "").replace("```", "")
    elif clean_text.startswith("```"):
         clean_text = clean_text.replace("```", "")
    
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        print(f"[Warning] Failed to parse JSON. Raw text: {clean_text[:50]}...")
        return {}

# ==================================================================================
# 2. THE AGENTIC PIPELINE
# ==================================================================================

def get_age_guidelines(age: int) -> Dict[str, str]:
    """
    Returns psychological and literary guidelines based on the child's age.
    """
    if age <= 6:
        return {
            "style": "Simple, repetitive, and rhythmic. Focus on clear cause-and-effect.",
            "vocabulary": "Concrete nouns (dog, ball) and action verbs (run, jump). Avoid abstract concepts.",
            "themes": "Friendship, sharing, daily routines, magical helpers, clear 'good vs bad'.",
            "complexity": "Linear plot. One main character. Happy, definite ending."
        }
    elif age <= 8:
        return {
            "style": "Engaging and descriptive. Start using longer sentences and some puns/humor.",
            "vocabulary": "Wider range, introduction of adverbs and adjectives. Simple figurative language.",
            "themes": "Empathy, courage, problem-solving, overcoming fear, school/social situations.",
            "complexity": "Character having a clear goal. Introduction of internal monologue/feelings."
        }
    else: # 9-10+
        return {
            "style": "Sophisticated and immersive. Use idioms, metaphors, and varied sentence structures.",
            "vocabulary": "Rich, specific, and abstract words (courage, betrayal, ancient, mysterious).",
            "themes": "Identity, loyalty, moral dilemmas, accepting differences, exploring the wider world.",
            "complexity": "Subplots allowed. Characters faced with tough choices. Personal growth is key."
        }

class PlannerAgent:
    """
    The Architect. Turns a vague idea into a structured 3-Act Arc.
    """
    def plan_story(self, user_request: str, age_guidelines: Dict[str, str]) -> Dict[str, str]:
        system_prompt = (
            "You are a world-class narrative architect for children's literature. "
            f"Target Audience Profile: {age_guidelines['complexity']}\n"
            f"Target Themes: {age_guidelines['themes']}\n"
            "Your goal is to design a captivating, original 3-Act story structure based on the user's request.\n"
            "GUIDELINES:\n"
            "- **Protagonist**: Give them a clear motivation and a distinct personality trait.\n"
            "- **Conflict**: Ensure there is a meaningful challenge that requires the protagonist to grow.\n"
            "- **Structure**: \n"
            "  - Setup: Introduce the status quo and the inciting incident.\n"
            "  - Confrontation: Rising action where obstacles get tougher.\n"
            "  - Resolution: A satisfying conclusion where the hero succeeds through their own effort.\n"
            "\n"
            "EXAMPLES OF GOOD PLANNING:\n"
            "Request: 'A mouse who wants to fly'\n"
            "Output: {\n"
            "  'reasoning': 'The theme of ambition vs limitation works well here. Conflict is physical inability. Resolution should be creative, not magic.',\n"
            "  'setup': 'Milo the mouse watches birds and builds wings from leaves.',\n"
            "  'confrontation': 'He tries to fly but crashes. The other mice laugh. A hawk chases him.',\n"
            "  'resolution': 'Milo uses his failed wings as a glider to escape the hawk, realizing he can glide if not fly.'\n"
            "}\n"
            "\n"
            "Output must be valid JSON with keys: 'reasoning', 'setup', 'confrontation', 'resolution'."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a story outline for: {user_request}"}
        ]
        
        print("Generating story arc...")
        response = call_model(messages, temperature=0.7, json_mode=True)
        return parse_json_output(response)

class StorytellerAgent:
    """
    The Artist. Writes the prose based on the plan.
    """
    def write_story(self, plan: Dict[str, str], age_guidelines: Dict[str, str], critique: Optional[str] = None) -> str:
        system_prompt = (
            "You are a master storyteller tailor-made for specific age groups. "
            f"Adhere to these STYLE GUIDELINES strictly:\n"
            f"- **Voice/Style**: {age_guidelines['style']}\n"
            f"- **Vocabulary Level**: {age_guidelines['vocabulary']}\n"
            "Write a story based STRICTLY on the provided outline.\n"
            "STORYTELLING RULES:\n"
            "1. **Show, Don't Tell**: Use sensory details suitable for the age group.\n"
            "2. **Pacing**: Keep it engaging.\n"
            "3. **Tone**: Adventurous, heartwarming, and safe.\n"
            "Length: 400-600 words."
        )
        
        user_content = (
            f"Here is the Narrative Plan:\n"
            f"1. Setup: {plan.get('setup')}\n"
            f"2. Confrontation: {plan.get('confrontation')}\n"
            f"3. Resolution: {plan.get('resolution')}\n"
        )
        
        if critique:
             user_content += (
                 f"\n\nIMPORTANT: The previous draft had issues. "
                 f"The Editor (Judge) provided this feedback: '{critique}' "
                 f"Refine the story to address these points specifically."
             )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        print("Writing story...")
        return call_model(messages, temperature=0.8) # Higher temp for creativity

class JudgeAgent:
    """
    The Critic. Evaluates safety, age-appropriateness, and quality.
    """
    def evaluate(self, story_text: str, age_guidelines: Dict[str, str]) -> Dict[str, Any]:
        system_prompt = (
            "You are a critical, discerning editor for a top-tier children's publisher. "
            "You generally only accept stories that are exceptional.\n"
            f"Target Audience Criteria:\n"
            f"- **Expected Vocabulary**: {age_guidelines['vocabulary']}\n"
            f"- **Expected Themes**: {age_guidelines['themes']}\n"
            "EVALUATION CRITERIA:\n"
            "1. **Safety**: (Pass/Fail) No violence, gore, scary themes, or inappropriate language.\n"
            "2. **Age Appropriateness**: Does it match the target profile above?\n"
            "3. **Show, Don't Tell**: Does the story use imagery and action rather than exposition?\n"
            "4. **Engagement**: Is the pacing good? Is the ending satisfying?\n"
            "\n"
            "Output valid JSON with keys:\n"
            "- 'thought_process': (string) Internal monologue analyzing the story step-by-step.\n"
            "- 'score': (integer 1-10)\n"
            "- 'feedback': (string). Be specific. If scoring < 8, explain exactly what to improve."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Story Text:\n{story_text}"}
        ]
        
        print("Critiquing draft...")
        response = call_model(messages, temperature=0.1, json_mode=True)
        # Fix: sometimes models might return just the json directly
        return parse_json_output(response)

# ==================================================================================
# 3. UTILITIES & MAIN
# ==================================================================================

def estimate_reading_time(text: str) -> str:
    word_count = len(text.split())
    minutes = math.ceil(word_count / 150) # Approx 150 wpm for read-aloud
    return f"{minutes} min read"

def extract_challenge_words(text: str) -> str:
    """
    Uses the LLM mainly as a quick utility to find 3 hard words and define them.
    """
    messages = [
        {"role": "user", "content": f"Identify 3 challenging words from this text for a 7-year-old and define them simply:\n\n{text}"}
    ]
    return call_model(messages, max_tokens=200)



    # 5. Future section as per original instructions
    """
    Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:
    
    1. **Audiobook Mode**: Integrate OpenAI's TTS (Text-to-Speech) API to read the story aloud in a soothing voice.
    2. **Illustrator Agent**: Use DALL-E 3 to generate unique cover art or scene illustrations based on the generated text.
    3. **Interactive "Choose Your Own Adventure"**: Pause the story after Act 2 and let the child decide the hero's next move.
    4. **Parent Dashboard**: A web UI to track stories, favorite words, and save the best ones to a library.
    """

def save_to_html(story_text: str, title: str, challenge_words: str):
    """
    Saves the story as a beautiful HTML file (The "Paperback" Edition).
    """
    filename = f"{title.replace(' ', '_').lower()}.html"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: 'Georgia', serif; padding: 40px; background: #fdf6e3; color: #333; max_width: 800px; margin: auto; line-height: 1.6; }}
            h1 {{ text-align: center; color: #2c3e50; font-size: 3em; margin-bottom: 20px; }}
            .story {{ font-size: 1.2em; white-space: pre-wrap; }}
            .extras {{ margin-top: 40px; padding: 20px; background: #eee8d5; border-radius: 10px; }}
            .footer {{ text-align: center; margin-top: 50px; font-size: 0.8em; color: #888; }}
        </style>
    </head>
    <body>
        <h1>{title.title()}</h1>
        <div class="story">{story_text}</div>
        <div class="extras">
            <h3>📚 For Little Learners</h3>
            <pre>{challenge_words}</pre>
        </div>
        <div class="footer">Generated by AI Bedtime Storyteller</div>
    </body>
    </html>
    """
    
    try:
        with open(filename, "w") as f:
            f.write(html_content)
        print(f"\n📖 Story saved to {filename} (Open it in your browser!)")
    except Exception as e:
        print(f"Could not save HTML: {e}")


# ==================================================================================
# 4. WEB SERVER & LOGIC
# ==================================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DreamHippo AI: Bedtime Stories</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Georgia&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Outfit', sans-serif; background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%); color: #e0e7ff; }
        .serif { font-family: 'Georgia', serif; }
        .glass { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .loader { border: 4px solid #f3f3f3; border-top: 4px solid #6366f1; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="min-h-screen flex items-center justify-center p-4">

    <!-- MAIN CONTAINER -->
    <div id="app" class="w-full max-w-4xl grid md:grid-cols-2 gap-8 items-start">
        
        <!-- INPUT FORM -->
        <div class="glass p-8 rounded-2xl shadow-2xl">
            <h1 class="text-3xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-indigo-300 to-purple-300">DreamHippo AI 🦛</h1>
            <p class="text-indigo-200 mb-6 text-sm">Create magical bedtime stories in seconds.</p>
            
            <form id="storyForm" class="space-y-4">
                <div>
                    <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-1">Story Topic</label>
                    <input type="text" id="topic" class="w-full bg-white/10 border border-indigo-500/30 rounded-lg p-3 text-white placeholder-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-400" placeholder="e.g. A brave toaster in space" required>
                </div>
                
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-1">Child's Name</label>
                        <input type="text" id="name" class="w-full bg-white/10 border border-indigo-500/30 rounded-lg p-3 text-white placeholder-indigo-400" placeholder="Optional">
                    </div>
                    <div>
                        <label class="block text-xs font-semibold uppercase tracking-wider text-indigo-300 mb-1">Age</label>
                        <input type="number" id="age" value="7" min="4" max="12" class="w-full bg-white/10 border border-indigo-500/30 rounded-lg p-3 text-white placeholder-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-400">
                    </div>
                </div>

                <button type="submit" class="w-full bg-gradient-to-r from-indigo-500 to-purple-600 hover:from-indigo-400 hover:to-purple-500 text-white font-bold py-3 rounded-lg transition-all transform hover:scale-[1.02] shadow-lg mt-6 flex items-center justify-center gap-2">
                    <span>✨ Weave Story</span>
                </button>
            </form>


        </div>

        <!-- OUTPUT DISPLAY -->
        <div class="glass p-8 rounded-2xl shadow-2xl min-h-[500px] flex flex-col relative">
            
            <!-- EMPTY STATE -->
            <div id="emptyState" class="flex-1 flex flex-col items-center justify-center text-center opacity-50">
                <div class="text-6xl mb-4">📖</div>
                <p class="text-indigo-200">Your story will appear here.</p>
            </div>

            <!-- LOADING STATE -->
            <div id="loadingState" class="hidden flex-1 flex flex-col items-center justify-center text-center">
                <div class="loader mb-4"></div>
                <p class="text-indigo-200 animate-pulse">Dreaming up a masterpiece...</p>
                <p class="text-xs text-indigo-400 mt-2">(Agents are planning, writing, and critiquing)</p>
            </div>

            <!-- RESULT STATE -->
            <div id="resultState" class="hidden flex-col h-full">
                <h2 id="storyTitle" class="text-2xl font-bold mb-4 text-amber-100 serif text-center"></h2>
                <div id="storyText" class="flex-1 overflow-y-auto pr-2 serif text-lg leading-relaxed text-indigo-50 mb-6 max-h-[400px]"></div>
                
                <div class="bg-indigo-900/50 p-4 rounded-lg mt-auto">
                    <h3 class="text-xs font-bold uppercase text-indigo-300 mb-2">📚 Challenge Words</h3>
                    <p id="challengeWords" class="text-sm text-indigo-100 whitespace-pre-wrap"></p>
                </div>
            </div>

        </div>
    </div>

    <script>
        document.getElementById('storyForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            // UI Update
            document.getElementById('emptyState').classList.add('hidden');
            document.getElementById('resultState').classList.add('hidden');
            document.getElementById('loadingState').classList.remove('hidden');
            
            const topic = document.getElementById('topic').value;
            const name = document.getElementById('name').value;
            const age = document.getElementById('age').value || 7;

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ topic, name, age })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Render Story
                document.getElementById('storyTitle').innerText = topic.toUpperCase();
                document.getElementById('storyText').innerText = data.story;
                document.getElementById('challengeWords').innerText = data.challenges;
                


                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('resultState').classList.remove('hidden');

            } catch (err) {
                console.error(err);
                alert('Something went wrong!');
                document.getElementById('loadingState').classList.add('hidden');
                document.getElementById('emptyState').classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
"""

def generate_story_logic(topic: str, child_name: str, age: int):
    """
    Core Logic Decoupled from CLI.
    """
    guidelines = get_age_guidelines(age)
    
    # Context enrichment
    full_request = f"Topic: {topic}."
    if child_name:
        full_request += f" Main character name: {child_name}."
    full_request += f" Audience Age: {age}."

    # Instantiate Agents
    planner = PlannerAgent()
    storyteller = StorytellerAgent()
    judge = JudgeAgent()
    
    # 1. Plan
    plan = planner.plan_story(full_request, guidelines)
    if not plan:
        return {"error": "Planning failed"}

    # 2. Write & Review Loop
    story_text = ""
    max_retries = 2
    attempts = 0
    passed = False
    critique = None
    
    while attempts <= max_retries and not passed:
        story_text = storyteller.write_story(plan, guidelines, critique)
        eval_result = judge.evaluate(story_text, guidelines)
        score = eval_result.get("score", 0)
        feedback = eval_result.get("feedback", "No feedback provided.")
        
        print(f"Judge Score: {score} | Feedback: {feedback}")
        
        if score >= 8:
            passed = True
        else:
            attempts += 1
            critique = feedback

    # 3. Extras
    reading_time = estimate_reading_time(story_text)
    challenges = extract_challenge_words(story_text)
    html_file = save_to_html(story_text, topic, challenges)
    
    return {
        "story": story_text,
        "challenges": challenges,
        "reading_time": reading_time,
        "html_file": html_file
    }

# Routes
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    result = generate_story_logic(
        data.get('topic'),
        data.get('name', ''),
        int(data.get('age', 7))
    )
    return jsonify(result)

def main():
    print("Starting AI Bedtime Story Server...")
    print("Open http://127.0.0.1:5001 in your browser to use the interface.")
    app.run(host='0.0.0.0', port=5001, debug=True)

if __name__ == "__main__":
    main()