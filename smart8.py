import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
import random
import google.generativeai as genai
import os

# ==========================================
# 1. KONFIGURASI
# ==========================================
GRID_SIZE = 10
BATTERY_MAX = 100
DRAIN_RATE = 1.0           
CLEANING_COST = 2.0        
RECHARGE_RATE = 20.0       
SAFETY_MARGIN = 20.0       

# Simulasi Bisnis
POWER_PRICE_PER_KWH = 0.15 
PANEL_OUTPUT_PER_TICK = 0.5 

# KODE ELEMEN
EMPTY = 0
WALL = 1
DIRT = 2
CHARGER = 3
OBSTACLE = 5    

# PALET WARNA NEON
COLOR_BG = '#0f172a'        
COLOR_GRID = '#1e293b'      
COLOR_ROBOT = '#06b6d4'     
COLOR_ROBOT_GLOW = '#22d3ee' 
COLOR_OBSTACLE = '#d946ef'  
COLOR_DIRT = '#facc15'      
COLOR_WALL = '#475569'      
COLOR_CHARGER = '#22c55e'   
COLOR_PATH = '#ef4444'      

# ==========================================
# 2. GEMINI AI INTEGRATION (SAFE MODE)
# ==========================================
def init_gemini():
    api_key = None
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif "GEMINI_API_KEY" in os.environ:
        api_key = os.environ["GEMINI_API_KEY"]

    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def get_ai_analysis(robot_state, grid_stats, weather):
    if not st.session_state.get('gemini_active', False):
        return "⚠️ AI Offline: API Key Missing."

    prompt = f"""
    Act as a Solar Farm Supervisor.
    Status: {robot_state['status']}
    Battery: {robot_state['battery']}%
    Efficiency: {grid_stats['efficiency']}%
    Environment: {weather if weather else "Clear"}
    
    Command: Give 1 short strategic order (Max 10 words).
    """

    try:
        # Pilihan 1: Coba model terbaru (Cepat & Murah)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        try:
            # Pilihan 2: Fallback ke model lama (Pasti ada)
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return "⚠️ AI System Rebooting..."

# ==========================================
# 3. LOGIKA AI (CORE SYSTEM)
# ==========================================
def bfs_shortest_path(grid, start, goal):
    rows, cols = grid.shape
    queue = deque([[start]])
    visited = set([start])
    if start == goal: return []

    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == goal: return path[1:] 

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] != WALL and grid[nr, nc] != OBSTACLE and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    new_path = list(path)
                    new_path.append((nr, nc))
                    queue.append(new_path)
    return None 

def find_nearest_dirt(grid, start):
    rows, cols = grid.shape
    dirt_locs = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == DIRT:
                dist = abs(start[0] - r) + abs(start[1] - c)
                dirt_locs.append(((r, c), dist))
    if not dirt_locs: return None
    dirt_locs.sort(key=lambda x: x[1])
    return dirt_locs[0][0]

def update_environment(grid, weather_event=None):
    rows, cols = grid.shape
    new_grid = grid.copy()
    
    if weather_event == "SANDSTORM":
        for _ in range(random.randint(5, 8)):
            r, c = random.randint(0, rows-1), random.randint(0, cols-1)
            if new_grid[r, c] == EMPTY: new_grid[r, c] = DIRT
    elif weather_event == "RAIN":
        for r in range(rows):
            for c in range(cols):
                if new_grid[r, c] == DIRT and random.random() < 0.4:
                    new_grid[r, c] = EMPTY

    obstacles = []
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == OBSTACLE:
                obstacles.append((r, c))
                new_grid[r, c] = EMPTY 
    
    for r, c in obstacles:
        moves = [(-1,0), (1,0), (0,-1), (0,1), (0,0)]
        dr, dc = random.choice(moves)
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            if new_grid[nr, nc] in [EMPTY, DIRT]:
                new_grid[nr, nc] = OBSTACLE
            else:
                new_grid[r, c] = OBSTACLE
        else:
            new_grid[r, c] = OBSTACLE
            
    return new_grid

# ==========================================
# 4. CLASS ROBOT
# ==========================================
class SmartRobot:
    def __init__(self, start_pos):
        self.pos = start_pos
        self.battery = BATTERY_MAX
        self.state = "IDLE" 
        self.total_cleaned = 0
        self.current_path = [] 
        self.status_msg = "System Online."
        self.ai_recommendation = "Initializing AI..." 

    def decide_and_move(self, grid):
        if self.battery <= 0 and self.pos != (0, 0):
            self.state = "DEAD"
            self.status_msg = "💀 BATTERY DEAD"
            self.battery = 0
            return

        if self.pos == (0, 0):
            if self.battery < BATTERY_MAX:
                self.state = "CHARGING"
                self.battery += RECHARGE_RATE
                if self.battery > BATTERY_MAX: self.battery = BATTERY_MAX
                self.status_msg = f"⚡ CHARGING... {int(self.battery)}%"
                return
            else:
                self.state = "IDLE"
                self.battery = BATTERY_MAX

        dist_to_home = abs(self.pos[0] - 0) + abs(self.pos[1] - 0)
        return_threshold = (dist_to_home * DRAIN_RATE) + SAFETY_MARGIN
        
        target = None
        if self.battery < return_threshold and self.pos != (0, 0):
            self.state = "RETURNING"
            target = (0, 0)
            self.status_msg = "⚠️ RETURNING TO BASE"
        else:
            target = find_nearest_dirt(grid, self.pos)
            if target:
                self.state = "CLEANING"
                self.status_msg = f"TARGET: {target}"
            else:
                self.state = "DONE"
                self.status_msg = "STANDING BY"
                target = (0, 0)

        if self.pos == target: return 

        path = bfs_shortest_path(grid, self.pos, target)
        self.current_path = path 

        if path:
            next_step = path[0]
            if grid[next_step[0], next_step[1]] == OBSTACLE:
                self.status_msg = "⛔ OBSTACLE - WAITING"
                return 

            self.pos = next_step
            self.battery = max(0, self.battery - DRAIN_RATE)
            
            r, c = next_step
            if grid[r, c] == DIRT:
                grid[r, c] = EMPTY
                self.total_cleaned += 1
                self.battery = max(0, self.battery - CLEANING_COST)
                self.status_msg = "✨ CLEANING"
        else:
            self.state = "BLOCKED"
            self.status_msg = "⚠️ PATH BLOCKED"

# ==========================================
# 5. UI & VISUALIZATION (SAFE MODE)
# ==========================================
def draw_visual_legend():
    """Menggambar legenda dengan metode Polygon Manual (Anti-Gagal)"""
    fig_leg, ax_leg = plt.subplots(figsize=(4, 2), facecolor=COLOR_BG)
    ax_leg.set_facecolor(COLOR_BG)
    ax_leg.axis('off')
    
    # 1. Robot (Lingkaran)
    ax_leg.add_patch(patches.Circle((0.1, 0.8), radius=0.05, color=COLOR_ROBOT))
    ax_leg.text(0.2, 0.78, "Robot (Agent)", color='white', fontsize=10)
    
    # 2. Obstacle (Segitiga Manual)
    # Kita gambar manual point-by-point agar tidak error versioning
    tri_x, tri_y = 0.1, 0.55
    r = 0.06
    triangle_points = [[tri_x, tri_y+r], [tri_x-r, tri_y-r], [tri_x+r, tri_y-r]]
    ax_leg.add_patch(patches.Polygon(triangle_points, color=COLOR_OBSTACLE))
    ax_leg.text(0.2, 0.53, "Dynamic Obstacle", color='white', fontsize=10)
    
    # 3. Dirt (Lingkaran Kecil)
    ax_leg.add_patch(patches.Circle((0.1, 0.3), radius=0.04, color=COLOR_DIRT))
    ax_leg.text(0.2, 0.28, "Dust/Debris", color='white', fontsize=10)
    
    # 4. Charger (Kotak)
    ax_leg.add_patch(patches.Rectangle((0.05, 0.05), width=0.1, height=0.1, color=COLOR_CHARGER, fill=False, linewidth=2))
    ax_leg.text(0.2, 0.08, "Charging Dock", color='white', fontsize=10)
    
    return fig_leg

# --- SETUP STREAMLIT ---
st.set_page_config(page_title="Solar Sentinel AI", layout="wide", page_icon="☀️")

st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_BG}; }}
    h1, h2, h3 {{ color: #38bdf8 !important; }}
    div[data-testid="stMetricValue"] {{ font-family: 'Courier New', monospace; color: #34d399; }}
    div[data-testid="stMarkdownContainer"] p {{ color: #cbd5e1; }}
</style>
""", unsafe_allow_html=True)

st.title("☀️ Solar Sentinel: AI-Powered Operations")
st.markdown("**Deployed on Vultr** | **Powered by Google Gemini**")

# Init State
if 'sim_map' not in st.session_state:
    st.session_state.sim_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    for _ in range(18): st.session_state.sim_map[random.randint(0,9), random.randint(0,9)] = DIRT
    for _ in range(6): st.session_state.sim_map[random.randint(0,9), random.randint(0,9)] = WALL
    for _ in range(3):
         while True:
            r,c = random.randint(0,9), random.randint(0,9)
            if st.session_state.sim_map[r,c]==EMPTY: st.session_state.sim_map[r,c]=OBSTACLE; break
    st.session_state.sim_map[0,0] = CHARGER
    st.session_state.bot = SmartRobot(start_pos=(0,0))
    st.session_state.run = False
    st.session_state.total_energy = 0.0
    st.session_state.weather_trigger = None
    st.session_state.gemini_active = init_gemini()
    st.session_state.last_ai_update = 0

col_ctrl, col_vis, col_stats = st.columns([1, 2, 1])

# --- CONTROL ---
with col_ctrl:
    st.subheader("🕹️ Command")
    if not st.session_state.run:
        if st.button("▶️ START MISSION", type="primary", use_container_width=True):
            st.session_state.run = True
            st.rerun()
    else:
        if st.button("⏸️ PAUSE", type="secondary", use_container_width=True):
            st.session_state.run = False
            st.rerun()

    st.divider()
    st.subheader("🌩️ Environment")
    c1, c2 = st.columns(2)
    if c1.button("🌪️ Storm"): st.session_state.weather_trigger = "SANDSTORM"
    if c2.button("🌧️ Rain"): st.session_state.weather_trigger = "RAIN"
    
    st.divider()
    st.subheader("ℹ️ Legend")
    st.pyplot(draw_visual_legend()) 

# --- AI & STATS ---
with col_stats:
    st.subheader("🧠 Gemini AI Copilot")
    ai_container = st.container(border=True)
    ai_container.markdown(f"**🤖 Supervisor:**\n\n*{st.session_state.bot.ai_recommendation}*")
    
    if not st.session_state.gemini_active:
        st.warning("⚠️ AI Offline (API Key Missing)")
    
    st.divider()
    st.subheader("📊 Telemetry")
    bot = st.session_state.bot
    
    dirt_count = np.count_nonzero(st.session_state.sim_map == DIRT)
    clean_cells = (GRID_SIZE * GRID_SIZE) - dirt_count
    efficiency = (clean_cells / (GRID_SIZE * GRID_SIZE)) * 100
    
    if st.session_state.run:
        produced = (efficiency / 100) * PANEL_OUTPUT_PER_TICK
        st.session_state.total_energy += produced
    
    revenue = st.session_state.total_energy * POWER_PRICE_PER_KWH

    st.metric("Grid Efficiency", f"{efficiency:.1f}%")
    st.metric("Power Generated", f"{st.session_state.total_energy:.2f} kWh")
    st.metric("Est. Revenue", f"${revenue:.2f}")
    
    st.metric("Battery", f"{int(bot.battery)}%")
    st.progress(int(bot.battery))
    st.code(f"STATUS: {bot.status_msg}")

# --- MAIN VISUALIZATION ---
with col_vis:
    fig, ax = plt.subplots(figsize=(6,6), facecolor=COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    
    ax.set_xticks(np.arange(-.5, GRID_SIZE, 1))
    ax.set_yticks(np.arange(-.5, GRID_SIZE, 1))
    ax.grid(color=COLOR_GRID, linestyle='-', linewidth=1.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    grid = st.session_state.sim_map
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell = grid[r, c]
            if cell == WALL:
                ax.plot([c-0.3, c+0.3], [r-0.3, r+0.3], color=COLOR_WALL, linewidth=3)
                ax.plot([c-0.3, c+0.3], [r+0.3, r-0.3], color=COLOR_WALL, linewidth=3)
            elif cell == CHARGER:
                rect = patches.Rectangle((c-.4, r-.4), 0.8, 0.8, linewidth=2, edgecolor=COLOR_CHARGER, facecolor='none')
                ax.add_patch(rect)
                ax.plot([c], [r], marker='+', color=COLOR_CHARGER, markersize=15, markeredgewidth=3)
            elif cell == DIRT:
                size = 0.2 + (r*c % 3)/10 
                circle = patches.Circle((c, r), size, color=COLOR_DIRT, alpha=0.9)
                ax.add_patch(circle)
            elif cell == OBSTACLE:
                # FIXED: Pakai Polygon Manual disini juga biar konsisten
                tri_pts = [[c, r+0.35], [c-0.35, r-0.35], [c+0.35, r-0.35]]
                ax.add_patch(patches.Polygon(tri_pts, color=COLOR_OBSTACLE))

    rr, rc = bot.pos
    glow = patches.Circle((rc, rr), 0.45, color=COLOR_ROBOT_GLOW, alpha=0.3)
    ax.add_patch(glow)
    core = patches.Circle((rc, rr), 0.25, color=COLOR_ROBOT, zorder=10)
    ax.add_patch(core)
    
    if bot.current_path:
        path_y = [p[0] for p in bot.current_path]
        path_x = [p[1] for p in bot.current_path]
        path_y.insert(0, rr); path_x.insert(0, rc)
        ax.plot(path_x, path_y, color=COLOR_PATH, linewidth=4, alpha=0.3)
        ax.plot(path_x, path_y, color=COLOR_PATH, linewidth=1.5, alpha=0.9)

    ax.set_xlim(-0.5, GRID_SIZE-0.5)
    ax.set_ylim(GRID_SIZE-0.5, -0.5)
    st.pyplot(fig)

# --- LOOP ---
if st.session_state.run or st.session_state.weather_trigger:
    time.sleep(0.2)
    w_event = st.session_state.weather_trigger
    st.session_state.sim_map = update_environment(st.session_state.sim_map, weather_event=w_event)
    st.session_state.weather_trigger = None
    
    if st.session_state.run:
        st.session_state.bot.decide_and_move(st.session_state.sim_map)
        
        if st.session_state.gemini_active:
            if time.time() - st.session_state.last_ai_update > 5: 
                stats = {
                    "efficiency": efficiency, 
                    "obstacles": np.count_nonzero(st.session_state.sim_map == OBSTACLE)
                }
                rob_state = {"battery": int(bot.battery), "status": bot.status_msg}
                st.session_state.bot.ai_recommendation = get_ai_analysis(rob_state, stats, w_event)
                st.session_state.last_ai_update = time.time()
        
        st.rerun()
