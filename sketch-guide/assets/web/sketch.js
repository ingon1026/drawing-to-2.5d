// Sketch Guide - Drawing + AI Guide Engine
// Based on magenta-demos predict.js, customized for guide mode

(function () {
  "use strict";

  // ===== Configuration =====
  var CONFIG = {
    guideColor: "rgba(74, 144, 217, 0.3)",      // semi-transparent blue
    guideLineWidth: 3.0,
    userColor: "#333333",
    userLineWidth: 2.5,
    temperature: 0.25,
    maxGuideSteps: 80,       // max predicted strokes for guide
    screenScaleFactor: 3.0,
    epsilon: 2.0,            // min distance to register a stroke
    modelBaseURL: "https://storage.googleapis.com/quickdraw-models/sketchRNN/large_models/",
    minSequenceLength: 5
  };

  // ===== State =====
  var canvas, ctx;
  var canvasWidth, canvasHeight;

  // Model
  var model = null;
  var modelState = null;
  var modelStateOrig = null;
  var currentCategory = "pig";
  var modelLoaded = false;

  // Drawing state
  var isDrawing = false;
  var hasStarted = false;
  var rawLines = [];          // array of simplified polylines
  var currentRawLine = [];    // current line being drawn
  var strokes = [];           // stroke-5 format for the model
  var startX, startY;         // starting position
  var lastX, lastY;           // last pen position

  // Guide state
  var guideStrokes = [];      // predicted guide strokes
  var guideVisible = false;

  // ===== Canvas Setup =====
  function initCanvas() {
    canvas = document.getElementById("drawing-canvas");
    ctx = canvas.getContext("2d");
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
  }

  function resizeCanvas() {
    canvasWidth = window.innerWidth;
    canvasHeight = window.innerHeight - 48 - 56; // minus status bar and toolbar
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    canvas.style.width = canvasWidth + "px";
    canvas.style.height = canvasHeight + "px";
    redrawAll();
  }

  // ===== Drawing Functions =====
  function drawLine(x1, y1, x2, y2, color, width) {
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  function drawStrokesOnCanvas(strokeArray, sx, sy, color, lineWidth) {
    var x = sx, y = sy;
    var prevPen = [1, 0, 0];

    for (var i = 0; i < strokeArray.length; i++) {
      var dx = strokeArray[i][0];
      var dy = strokeArray[i][1];
      var penDown = strokeArray[i][2];
      var penUp = strokeArray[i][3];
      var penEnd = strokeArray[i][4];

      if (prevPen[2] === 1) break; // end of drawing

      if (prevPen[0] === 1) {
        drawLine(x, y, x + dx, y + dy, color, lineWidth);
      }

      x += dx;
      y += dy;
      prevPen = [penDown, penUp, penEnd];
    }
  }

  function redrawAll() {
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Draw user strokes
    if (strokes.length > 0) {
      drawStrokesOnCanvas(strokes, startX, startY, CONFIG.userColor, CONFIG.userLineWidth);
    }

    // Draw guide strokes
    if (guideVisible && guideStrokes.length > 0) {
      var guideStartX, guideStartY;
      if (rawLines.length > 0) {
        var lastLine = rawLines[rawLines.length - 1];
        var lastPt = lastLine[lastLine.length - 1];
        guideStartX = lastPt[0];
        guideStartY = lastPt[1];
      } else {
        guideStartX = startX;
        guideStartY = startY;
      }
      drawStrokesOnCanvas(guideStrokes, guideStartX, guideStartY, CONFIG.guideColor, CONFIG.guideLineWidth);
    }
  }

  // ===== Pointer Event Handlers =====
  function getPointerPos(e) {
    var rect = canvas.getBoundingClientRect();
    return {
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    };
  }

  function onPointerDown(e) {
    e.preventDefault();
    isDrawing = true;
    var pos = getPointerPos(e);

    if (!hasStarted) {
      hasStarted = true;
      startX = pos.x;
      startY = pos.y;
      lastX = pos.x;
      lastY = pos.y;
    }

    lastX = pos.x;
    lastY = pos.y;
    currentRawLine = [[pos.x, pos.y]];

    // Hide guide while drawing
    guideVisible = false;
    redrawAll();
  }

  function onPointerMove(e) {
    e.preventDefault();
    if (!isDrawing) return;

    var pos = getPointerPos(e);
    var dx = pos.x - lastX;
    var dy = pos.y - lastY;

    if (dx * dx + dy * dy > CONFIG.epsilon * CONFIG.epsilon) {
      drawLine(lastX, lastY, pos.x, pos.y, CONFIG.userColor, CONFIG.userLineWidth);
      lastX = pos.x;
      lastY = pos.y;
      currentRawLine.push([pos.x, pos.y]);
    }
  }

  function onPointerUp(e) {
    e.preventDefault();
    if (!isDrawing) return;
    isDrawing = false;

    if (currentRawLine.length < 2) {
      currentRawLine = [];
      return;
    }

    // Simplify the line
    var simplifiedLine = DataTool.simplify_line(currentRawLine);

    if (simplifiedLine.length > 1) {
      // Get last point for reference
      var refX, refY;
      if (rawLines.length === 0) {
        refX = startX;
        refY = startY;
      } else {
        var idx = rawLines.length - 1;
        var lastPt = rawLines[idx][rawLines[idx].length - 1];
        refX = lastPt[0];
        refY = lastPt[1];
      }

      // Convert to stroke-5 format
      var newStroke = DataTool.line_to_stroke(simplifiedLine, [refX, refY]);
      rawLines.push(simplifiedLine);
      strokes = strokes.concat(newStroke);

      // Redraw with simplified strokes
      redrawAll();

      // Generate guide
      if (modelLoaded && strokes.length >= CONFIG.minSequenceLength) {
        generateGuide();
        setStatus("Guide is shown - keep drawing!");
      } else if (modelLoaded) {
        setStatus("Keep drawing... (" + strokes.length + "/" + CONFIG.minSequenceLength + " strokes)");
      }
    }

    currentRawLine = [];
  }

  // ===== Guide Generation =====
  function generateGuide() {
    if (!model || strokes.length === 0) return;

    // Encode user strokes into model state
    modelStateOrig = model.zero_state();
    modelStateOrig = model.update(model.zero_input(), modelStateOrig);

    for (var i = 0; i < strokes.length - 1; i++) {
      modelStateOrig = model.update(strokes[i], modelStateOrig);
    }

    // Get the last stroke to initialize prediction
    var lastStroke = strokes[strokes.length - 1];
    var state = model.copy_state(modelStateOrig);

    var dx = lastStroke[0];
    var dy = lastStroke[1];
    var penDown = lastStroke[2];
    var penUp = lastStroke[3];
    var penEnd = lastStroke[4];

    // Predict next strokes
    guideStrokes = [];
    for (var step = 0; step < CONFIG.maxGuideSteps; step++) {
      state = model.update([dx, dy, penDown, penUp, penEnd], state);
      var pdf = model.get_pdf(state);
      var sample = model.sample(pdf, CONFIG.temperature);

      dx = sample[0];
      dy = sample[1];
      penDown = sample[2];
      penUp = sample[3];
      penEnd = sample[4];

      guideStrokes.push([dx, dy, penDown, penUp, penEnd]);

      if (penEnd === 1) break;
    }

    guideVisible = true;
    redrawAll();
  }

  // ===== Model Loading =====
  function loadModel(category) {
    currentCategory = category;
    modelLoaded = false;
    showLoading("Loading " + category + " model...");
    setModelLabel(category);

    var url = CONFIG.modelBaseURL + category + ".gen.json";

    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, true);
    xhr.onload = function () {
      if (xhr.status === 200) {
        try {
          var modelData = JSON.parse(xhr.responseText);
          model = new SketchRNN(modelData);
          model.set_pixel_factor(CONFIG.screenScaleFactor);
          modelLoaded = true;
          hideLoading();
          setStatus("Draw a " + category + "!");
          notifyNative({ event: "modelLoaded", category: category });
        } catch (e) {
          hideLoading();
          setStatus("Error loading model");
          console.error("Model parse error:", e);
        }
      } else {
        hideLoading();
        setStatus("Failed to load model");
      }
    };
    xhr.onerror = function () {
      hideLoading();
      setStatus("Network error loading model");
    };
    xhr.send();
  }

  // ===== Public API (called from React Native) =====
  window.clearDrawing = function () {
    hasStarted = false;
    rawLines = [];
    currentRawLine = [];
    strokes = [];
    guideStrokes = [];
    guideVisible = false;
    lastX = 0;
    lastY = 0;
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);
    setStatus("Draw a " + currentCategory + "!");
  };

  window.undoStroke = function () {
    if (rawLines.length === 0) return;

    rawLines.pop();

    // Rebuild strokes from rawLines
    strokes = [];
    for (var i = 0; i < rawLines.length; i++) {
      var refX, refY;
      if (i === 0) {
        refX = startX;
        refY = startY;
      } else {
        var prevLine = rawLines[i - 1];
        var lastPt = prevLine[prevLine.length - 1];
        refX = lastPt[0];
        refY = lastPt[1];
      }
      var stroke = DataTool.line_to_stroke(rawLines[i], [refX, refY]);
      strokes = strokes.concat(stroke);
    }

    // Regenerate guide
    guideStrokes = [];
    guideVisible = false;
    if (modelLoaded && strokes.length >= CONFIG.minSequenceLength) {
      generateGuide();
    }

    redrawAll();

    if (rawLines.length === 0) {
      hasStarted = false;
      setStatus("Draw a " + currentCategory + "!");
    }
  };

  window.setTemperature = function (value) {
    CONFIG.temperature = Math.max(0.1, Math.min(1.0, value));
    if (guideVisible && strokes.length >= CONFIG.minSequenceLength) {
      generateGuide();
    }
  };

  // ===== UI Helpers =====
  function setStatus(text) {
    document.getElementById("status-text").textContent = text;
  }

  function setModelLabel(name) {
    document.getElementById("model-label").textContent = name;
  }

  function showLoading(text) {
    document.getElementById("loading-text").textContent = text;
    document.getElementById("loading-overlay").classList.remove("hidden");
  }

  function hideLoading() {
    document.getElementById("loading-overlay").classList.add("hidden");
  }

  // ===== React Native Communication =====
  function notifyNative(data) {
    if (window.ReactNativeWebView) {
      window.ReactNativeWebView.postMessage(JSON.stringify(data));
    }
  }

  // Listen for messages from React Native
  window.addEventListener("message", function (e) {
    try {
      var msg = typeof e.data === "string" ? JSON.parse(e.data) : e.data;
      switch (msg.action) {
        case "loadModel":
          loadModel(msg.category);
          break;
        case "clear":
          window.clearDrawing();
          break;
        case "setTemperature":
          window.setTemperature(msg.value);
          break;
      }
    } catch (err) {
      console.error("Message parse error:", err);
    }
  });

  // Also listen for document message (Android WebView compatibility)
  document.addEventListener("message", function (e) {
    try {
      var msg = typeof e.data === "string" ? JSON.parse(e.data) : e.data;
      switch (msg.action) {
        case "loadModel":
          loadModel(msg.category);
          break;
        case "clear":
          window.clearDrawing();
          break;
        case "setTemperature":
          window.setTemperature(msg.value);
          break;
      }
    } catch (err) {
      console.error("Message parse error:", err);
    }
  });

  // ===== Init =====
  function init() {
    initCanvas();

    // Attach pointer events
    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointerleave", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);

    // Prevent default touch behaviors on canvas
    canvas.addEventListener("touchstart", function (e) { e.preventDefault(); }, { passive: false });
    canvas.addEventListener("touchmove", function (e) { e.preventDefault(); }, { passive: false });

    // Load default model
    loadModel(currentCategory);
  }

  // Wait for DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
