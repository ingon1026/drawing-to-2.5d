#!/usr/bin/env node
const fs = require('fs');
const path = require('path');

const modelsDir = path.join(__dirname, '../assets/models');
const libDir = path.join(__dirname, '../assets/web/lib');
// SketchRNN 모델 카테고리 (large 우선, 없으면 small fallback)
const categories = ['pig'];

console.log('Loading libraries...');
const numjsCode = fs.readFileSync(path.join(libDir, 'numjs.js'), 'utf8');
const sketchRNNCode = fs.readFileSync(path.join(libDir, 'sketch_rnn.js'), 'utf8');

console.log('Loading models...');
const modelEntries = {};
for (const cat of categories) {
  let filePath = path.join(modelsDir, `${cat}.large.gen.json`);
  if (!fs.existsSync(filePath)) filePath = path.join(modelsDir, `${cat}.gen.json`);
  const data = fs.readFileSync(filePath, 'utf8');
  modelEntries[cat] = data;
  console.log(`  ${cat}: ${(data.length / 1024 / 1024).toFixed(1)} MB`);
}

const modelDataScript = `var LOCAL_MODELS = {};
${Object.entries(modelEntries).map(([cat, json]) =>
  `LOCAL_MODELS["${cat}"] = ${json};`
).join('\n')}`;

const html = `<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <meta name="robots" content="noindex, nofollow">
  <meta name="description" content="AI와 함께 그리는 드로잉 가이드 — K3I 사내 도구">
  <title>Sketch Guide — K3I</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    html, body {
      width: 100%; height: 100%;
      overflow: hidden; touch-action: none;
      background: #fff; font-family: 'Segoe UI', sans-serif;
    }
    #status-bar {
      position: fixed; top: 0; left: 0; right: 0; height: 48px;
      background: #f8f9fa; border-bottom: 1px solid #e0e0e0;
      display: flex; align-items: center; justify-content: center; gap: 12px;
      font-size: 14px; color: #333; z-index: 10;
    }
    .model-name { font-weight: 600; color: #4A90D9; font-size: 16px; }
    .status-text { font-size: 13px; color: #999; }
    .state-dot {
      width: 10px; height: 10px; border-radius: 50%;
      background: #ccc; transition: background 0.3s;
    }
    .state-dot.drawing { background: #4CAF50; }
    .state-dot.waiting { background: #FF9800; }
    .state-dot.animating { background: #4A90D9; animation: pulse 0.8s infinite; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
    #drawing-canvas { position: fixed; top: 48px; left: 0; cursor: crosshair; }
    #toolbar {
      position: fixed; bottom: 0; left: 0; right: 0; height: 56px;
      background: #f8f9fa; border-top: 1px solid #e0e0e0;
      display: flex; align-items: center; justify-content: center; gap: 10px;
      z-index: 10; padding: 4px 8px;
    }
    .tool-btn {
      padding: 8px 18px; border: 1px solid #ddd; border-radius: 8px;
      background: white; font-size: 14px; cursor: pointer;
      -webkit-tap-highlight-color: transparent;
    }
    .tool-btn:active { background: #e8e8e8; }
    select.tool-btn { padding: 6px 12px; }
    #loading-overlay {
      position: fixed; top: 0; left: 0; right: 0; bottom: 0;
      background: rgba(255,255,255,0.92);
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      font-size: 18px; color: #666; gap: 8px; z-index: 100;
    }
    #loading-overlay.hidden { display: none; }
    #side-toast {
      position: fixed; right: 16px; top: 70px;
      background: white;
      border: 1px solid #e0e0e0;
      border-left: 4px solid #4CAF50;
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 4px 20px rgba(0,0,0,0.12);
      display: flex; align-items: center; gap: 10px;
      z-index: 50;
      transform: translateX(0);
      transition: transform 0.3s ease, opacity 0.3s ease;
      opacity: 1;
    }
    #side-toast.toast-hidden {
      transform: translateX(calc(100% + 20px));
      opacity: 0;
      pointer-events: none;
    }
    .toast-msg { font-size: 14px; color: #333; font-weight: 500; white-space: nowrap; }
    .toast-btn {
      padding: 6px 16px; border: none; border-radius: 8px;
      background: #4CAF50; color: white; font-size: 13px;
      font-weight: 600; cursor: pointer; white-space: nowrap;
    }
    .toast-btn:active { background: #388E3C; }
    .toast-dismiss {
      background: none; border: none; font-size: 18px;
      color: #aaa; cursor: pointer; padding: 0 4px;
    }
  </style>
</head>
<body>
  <div id="status-bar">
    <div class="state-dot" id="state-dot"></div>
    <span class="model-name" id="model-label">pig</span>
    <span class="status-text" id="status-text">Loading...</span>
  </div>
  <canvas id="drawing-canvas"></canvas>
  <div id="toolbar">
    <select class="tool-btn" id="category-select" onchange="switchCategory(this.value)">
      ${categories.map(c => `<option value="${c}">${c.charAt(0).toUpperCase() + c.slice(1)}</option>`).join('\n      ')}
    </select>
    <button class="tool-btn" onclick="clearDrawing()">Clear</button>
    <button class="tool-btn" onclick="undoStroke()">Undo</button>
    <label style="font-size:12px;color:#666;">
      T: <input type="range" id="temp-slider" min="10" max="80" value="25"
             oninput="updateTemp(this.value)" style="width:70px;vertical-align:middle;">
      <span id="temp-val">25</span>
    </label>
  </div>
  <div id="loading-overlay">
    <span id="loading-text">Loading...</span>
  </div>
  <div id="side-toast" class="toast-hidden">
    <div class="toast-msg">Nice! Done drawing?</div>
    <button class="toast-btn" onclick="confirmComplete()">Done!</button>
    <button class="toast-dismiss" onclick="dismissToast()">&times;</button>
  </div>

  <script>${numjsCode}</script>
  <script>${sketchRNNCode}</script>
  <script>${modelDataScript}</script>
  <script>
  (function () {
    "use strict";

    // ===== CONFIG =====
    var temperature = 0.25;
    var SCALE_FACTOR = 3.0;
    var EPSILON = 2.0;
    var MIN_SEQ = 3;
    var WAIT_MS = 400;           // ms to wait after pen up before AI starts
    var LOOP_PAUSE_MS = 600;     // ms pause between AI loop variations
    var MAX_LOOPS = 3;           // stop after N variations
    var COMPLETE_THRESHOLD = 15; // if AI predicts < this many strokes, drawing is "done"
    var shortPredictions = 0;    // count how many loops ended quickly
    var USER_COLOR = "#333333";
    var USER_LINE_W = 2.5;
    var AI_LINE_W = 3.0;

    // Rotating guide colors for each loop variation
    var AI_COLORS = [
      "rgba(74, 144, 217, 0.45)",
      "rgba(217, 74, 144, 0.40)",
      "rgba(74, 217, 144, 0.40)",
      "rgba(200, 150, 50, 0.40)",
      "rgba(150, 74, 217, 0.40)"
    ];
    var aiColorIdx = 0;

    // ===== STATE MACHINE =====
    // IDLE: nothing happening
    // USER_DRAWING: pen is down, user is drawing
    // WAITING: pen lifted, waiting to see if user draws more
    // AI_ANIMATING: AI is drawing prediction frame by frame
    var appState = "IDLE";

    // ===== CANVAS =====
    var canvas, ctx, canvasW, canvasH;

    // ===== MODEL =====
    var model = null;
    var currentCategory = "pig";
    var modelLoaded = false;

    // ===== USER DRAWING STATE =====
    var hasStarted = false;
    var rawLines = [];
    var currentRawLine = [];
    var strokes = [];
    var startX, startY, lastX, lastY;

    // ===== AI STATE =====
    var waitTimer = null;
    var aiAnimId = null;
    var aiState = null;         // model RNN state for AI
    var aiStateOrig = null;     // saved encoded state for looping
    var aiLastStroke = null;    // last stroke fed to model
    var aiX, aiY;               // AI pen position
    var aiDx, aiDy, aiP0, aiP1, aiP2;
    var aiPrevPen;
    var currentAIColor;

    // ===== INIT =====
    function initCanvas() {
      canvas = document.getElementById("drawing-canvas");
      ctx = canvas.getContext("2d");
      resizeCanvas();
      window.addEventListener("resize", resizeCanvas);
    }

    function resizeCanvas() {
      canvasW = window.innerWidth;
      canvasH = window.innerHeight - 48 - 56;
      canvas.width = canvasW;
      canvas.height = canvasH;
      canvas.style.width = canvasW + "px";
      canvas.style.height = canvasH + "px";
      redrawUserStrokes();
    }

    // ===== DRAWING HELPERS =====
    function drawLine(x1, y1, x2, y2, color, w) {
      ctx.beginPath();
      ctx.moveTo(x1, y1); ctx.lineTo(x2, y2);
      ctx.strokeStyle = color; ctx.lineWidth = w;
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.stroke();
    }

    function redrawUserStrokes() {
      ctx.clearRect(0, 0, canvasW, canvasH);
      if (strokes.length === 0) return;
      var x = startX, y = startY, prev = [1, 0, 0];
      for (var i = 0; i < strokes.length; i++) {
        if (prev[2] === 1) break;
        if (prev[0] === 1)
          drawLine(x, y, x + strokes[i][0], y + strokes[i][1], USER_COLOR, USER_LINE_W);
        x += strokes[i][0]; y += strokes[i][1];
        prev = [strokes[i][2], strokes[i][3], strokes[i][4]];
      }
    }

    // ===== POINTER EVENTS =====
    function getPos(e) {
      var r = canvas.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    }

    function onDown(e) {
      e.preventDefault();

      // Interrupt AI if it's animating
      if (appState === "AI_ANIMATING") {
        stopAI();
        redrawUserStrokes();
      }
      // Hide toast if visible
      dismissToast();

      // Cancel any pending wait timer
      clearTimeout(waitTimer);

      appState = "USER_DRAWING";
      updateStateDot();

      var p = getPos(e);
      if (!hasStarted) { hasStarted = true; startX = p.x; startY = p.y; }
      lastX = p.x; lastY = p.y;
      currentRawLine = [[p.x, p.y]];
    }

    function onMove(e) {
      e.preventDefault();
      if (appState !== "USER_DRAWING") return;
      var p = getPos(e), dx = p.x - lastX, dy = p.y - lastY;
      if (dx * dx + dy * dy > EPSILON * EPSILON) {
        drawLine(lastX, lastY, p.x, p.y, USER_COLOR, USER_LINE_W);
        lastX = p.x; lastY = p.y;
        currentRawLine.push([p.x, p.y]);
      }
    }

    function onUp(e) {
      e.preventDefault();
      if (appState !== "USER_DRAWING") return;

      // Process the completed line
      if (currentRawLine.length >= 2) {
        var sl = DataTool.simplify_line(currentRawLine);
        if (sl.length > 1) {
          var rx, ry;
          if (rawLines.length === 0) { rx = startX; ry = startY; }
          else { var lp = rawLines[rawLines.length-1]; var pp = lp[lp.length-1]; rx = pp[0]; ry = pp[1]; }
          var ns = DataTool.line_to_stroke(sl, [rx, ry]);
          rawLines.push(sl);
          strokes = strokes.concat(ns);
          // Redraw clean version
          redrawUserStrokes();
        }
      }
      currentRawLine = [];

      // Enter WAITING state - debounce before starting AI
      appState = "WAITING";
      updateStateDot();
      setStatus("...");

      clearTimeout(waitTimer);
      waitTimer = setTimeout(function() {
        if (appState !== "WAITING") return;
        if (modelLoaded && strokes.length >= MIN_SEQ) {
          encodeAndStartAI();
        } else if (modelLoaded) {
          appState = "IDLE";
          updateStateDot();
          setStatus("Keep drawing... (" + strokes.length + "/" + MIN_SEQ + ")");
        }
      }, WAIT_MS);
    }

    // ===== ENCODING =====
    function encodeAndStartAI() {
      setStatus("Encoding...");

      // Feed all user strokes into model RNN
      aiStateOrig = model.zero_state();
      aiStateOrig = model.update(model.zero_input(), aiStateOrig);
      for (var i = 0; i < strokes.length - 1; i++) {
        aiStateOrig = model.update(strokes[i], aiStateOrig);
      }
      aiLastStroke = strokes[strokes.length - 1];

      // Reset for new encoding
      aiColorIdx = 0;
      shortPredictions = 0;

      // Start first AI animation
      startAIAnimation();
    }

    // ===== AI ANIMATION =====
    var aiStepCount = 0; // count strokes in current prediction

    function startAIAnimation() {
      appState = "AI_ANIMATING";
      updateStateDot();
      aiStepCount = 0;

      // Copy encoded state (so we can loop from same point)
      aiState = model.copy_state(aiStateOrig);

      // Initialize AI pen position at user's last point
      if (rawLines.length > 0) {
        var ll = rawLines[rawLines.length - 1];
        var lp = ll[ll.length - 1];
        aiX = lp[0]; aiY = lp[1];
      } else {
        aiX = startX; aiY = startY;
      }

      // Start from last user stroke
      aiDx = aiLastStroke[0];
      aiDy = aiLastStroke[1];
      aiP0 = aiLastStroke[2];
      aiP1 = aiLastStroke[3];
      aiP2 = aiLastStroke[4];
      aiPrevPen = [1, 0, 0];

      // Pick color for this variation
      currentAIColor = AI_COLORS[aiColorIdx % AI_COLORS.length];

      setStatus("AI is drawing...");
      aiFrame();
    }

    function aiFrame() {
      if (appState !== "AI_ANIMATING") return;

      // Update model state and sample next point
      aiState = model.update([aiDx, aiDy, aiP0, aiP1, aiP2], aiState);
      var pdf = model.get_pdf(aiState);
      var s = model.sample(pdf, temperature);

      aiDx = s[0]; aiDy = s[1];
      aiP0 = s[2]; aiP1 = s[3]; aiP2 = s[4];

      aiStepCount++;

      // Draw if pen is down
      if (aiPrevPen[0] === 1) {
        drawLine(aiX, aiY, aiX + aiDx, aiY + aiDy, currentAIColor, AI_LINE_W);
      }

      aiX += aiDx;
      aiY += aiDy;
      aiPrevPen = [aiP0, aiP1, aiP2];

      // Check if AI prediction ended
      if (aiP2 === 1) {
        // Was this a short prediction? → model thinks drawing is nearly complete
        if (aiStepCount < COMPLETE_THRESHOLD) {
          shortPredictions++;
        }

        // If 2+ predictions were short → drawing is complete!
        if (shortPredictions >= 2) {
          showComplete();
          return;
        }

        // Otherwise loop with next variation
        setStatus("Variation " + (aiColorIdx+1) + "/" + MAX_LOOPS);
        setTimeout(function() {
          if (appState !== "AI_ANIMATING") return;
          redrawUserStrokes();
          aiColorIdx++;
          if (aiColorIdx >= MAX_LOOPS) {
            // Max loops reached without completion
            appState = "IDLE";
            updateStateDot();
            setStatus("Draw more or clear to restart.");
            return;
          }
          startAIAnimation();
        }, LOOP_PAUSE_MS);
        return;
      }

      aiAnimId = requestAnimationFrame(aiFrame);
    }

    function stopAI() {
      appState = "IDLE";
      updateStateDot();
      cancelAnimationFrame(aiAnimId);
      clearTimeout(waitTimer);
      setStatus("Draw more or wait for AI");
    }

    var toastTimer = null;

    function showComplete() {
      stopAI();
      redrawUserStrokes();
      document.getElementById("side-toast").classList.remove("toast-hidden");
      setStatus("Looks complete!");
      // Auto-hide after 6 seconds if not interacted
      clearTimeout(toastTimer);
      toastTimer = setTimeout(dismissToast, 6000);
    }

    window.dismissToast = function() {
      document.getElementById("side-toast").classList.add("toast-hidden");
      clearTimeout(toastTimer);
    };

    window.confirmComplete = function() {
      dismissToast();
      setStatus("Great job! Clear to draw something new.");
    };

    // ===== MODEL LOADING =====
    function loadModel(category) {
      currentCategory = category;
      modelLoaded = false;
      showLoading("Loading " + category + "...");
      setModelLabel(category);
      setTimeout(function() {
        try {
          var data = LOCAL_MODELS[category];
          if (!data) { hideLoading(); setStatus("Not found: " + category); return; }
          model = new SketchRNN(data);
          model.set_pixel_factor(SCALE_FACTOR);
          modelLoaded = true;
          hideLoading();
          setStatus("Draw a " + category + "!");
          appState = "IDLE";
          updateStateDot();
        } catch(e) { hideLoading(); setStatus("Error: " + e.message); }
      }, 50);
    }

    // ===== PUBLIC API =====
    window.clearDrawing = function() {
      stopAI();
      hasStarted = false; rawLines = []; currentRawLine = []; strokes = [];
      aiStateOrig = null; aiLastStroke = null;
      ctx.clearRect(0, 0, canvasW, canvasH);
      setStatus("Draw a " + currentCategory + "!");
    };

    window.undoStroke = function() {
      stopAI();
      if (rawLines.length === 0) return;
      rawLines.pop(); strokes = [];
      for (var i = 0; i < rawLines.length; i++) {
        var rx, ry;
        if (i === 0) { rx = startX; ry = startY; }
        else { var pl = rawLines[i-1]; var pp = pl[pl.length-1]; rx = pp[0]; ry = pp[1]; }
        strokes = strokes.concat(DataTool.line_to_stroke(rawLines[i], [rx, ry]));
      }
      redrawUserStrokes();
      if (rawLines.length === 0) { hasStarted = false; setStatus("Draw a " + currentCategory + "!"); }
      else setStatus("Stroke removed. Draw more or wait for AI.");
    };

    window.switchCategory = function(cat) { window.clearDrawing(); loadModel(cat); };

    window.updateTemp = function(v) {
      temperature = Math.max(0.05, Math.min(0.8, v / 100));
      document.getElementById("temp-val").textContent = v;
    };

    // ===== UI =====
    function setStatus(t) { document.getElementById("status-text").textContent = t; }
    function setModelLabel(n) { document.getElementById("model-label").textContent = n; }
    function showLoading(t) {
      document.getElementById("loading-text").textContent = t;
      document.getElementById("loading-overlay").classList.remove("hidden");
    }
    function hideLoading() { document.getElementById("loading-overlay").classList.add("hidden"); }

    function updateStateDot() {
      var dot = document.getElementById("state-dot");
      dot.className = "state-dot";
      if (appState === "USER_DRAWING") dot.classList.add("drawing");
      else if (appState === "WAITING") dot.classList.add("waiting");
      else if (appState === "AI_ANIMATING") dot.classList.add("animating");
    }

    // ===== INIT =====
    function init() {
      initCanvas();
      canvas.addEventListener("pointerdown", onDown);
      canvas.addEventListener("pointermove", onMove);
      canvas.addEventListener("pointerup", onUp);
      canvas.addEventListener("pointerleave", onUp);
      canvas.addEventListener("pointercancel", onUp);
      canvas.addEventListener("touchstart", function(e){e.preventDefault();}, {passive:false});
      canvas.addEventListener("touchmove", function(e){e.preventDefault();}, {passive:false});
      loadModel(currentCategory);
    }

    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  })();
  </script>
</body>
</html>`;

const outputPath = path.join(__dirname, '../standalone.html');
fs.writeFileSync(outputPath, html, 'utf8');
const sizeMB = (fs.statSync(outputPath).size / 1024 / 1024).toFixed(1);
console.log('\\n✅ standalone.html (' + sizeMB + ' MB)');
console.log('   Models: ' + categories.join(', '));
