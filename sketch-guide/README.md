# Sketch Guide

Real-time AI drawing assistant that guides users as they sketch. When a user starts drawing (e.g., a pig), the AI watches the strokes and animates what to draw next — like having a drawing tutor looking over your shoulder.

Built on [Google Magenta's Sketch RNN](https://magenta.tensorflow.org/sketch-rnn-demo), trained on millions of drawings from the [Quick Draw dataset](https://quickdraw.withgoogle.com/data).

## Demo

```
User draws a few strokes  →  AI animates the rest in real-time
                            (semi-transparent blue lines)
```

1. Pick a category (pig, cat, dog, bird, flower)
2. Start drawing on the canvas
3. Lift your pen — AI begins animating a suggested completion
4. Follow the guide, or keep drawing your own way
5. AI adapts to your new strokes each time you pause

## How It Works

```
User draws strokes
    ↓
Pen lifted → 400ms debounce (waits for more strokes)
    ↓
Line simplification (Ramer-Douglas-Peucker) → stroke-5 encoding
    ↓
Feed all strokes into Sketch RNN encoder (one-time)
    ↓
AI samples next points frame-by-frame → animates on canvas
    ↓
On completion → shows up to 3 variations in different colors
    ↓
If drawing looks complete → sidebar prompt: "Looks done?"
    ↓
User touches canvas → AI stops instantly, new cycle begins
```

### Key Design Decisions

- **Debounced encoding**: AI waits 400ms after pen-up before starting, so rapid multi-stroke drawing isn't interrupted
- **Frame-by-frame animation**: AI draws one point per frame (not a static dump), making predictions feel natural
- **Variation loop**: After completing a prediction, AI shows alternative completions (up to 3) with different colors
- **Completion detection**: When the model's predictions become very short (< 15 strokes) twice in a row, it means the drawing is nearly complete

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Drawing model | Sketch RNN (LSTM VAE), Google Magenta |
| Training data | Google Quick Draw dataset |
| Stroke format | stroke-5: `[dx, dy, pen_down, pen_up, pen_end]` |
| Rendering | HTML5 Canvas + PointerEvents |
| Bundling | Single self-contained HTML file (all JS + models inlined) |
| Mobile | Capacitor → Android APK |

## Quick Start

### 1. Download models

```bash
cd sketch-guide
node scripts/download-models.js
```

Downloads pre-trained Sketch RNN models (~12MB each) from Google Cloud Storage.

### 2. Build

```bash
node scripts/build-standalone.js
```

Generates `standalone.html` — a single file containing all libraries, models, and the drawing engine. No server or internet required to run.

### 3. Run

```bash
# Linux / WSL2 (opens Chrome in app mode)
./run.sh

# Or just open standalone.html in any browser
```

### Android APK (optional)

See [Building APK](#building-apk) below.

## Project Structure

```
sketch-guide/
├── standalone.html            ← generated (gitignored)
├── run.sh                     ← launch script
├── package.json
├── scripts/
│   ├── build-standalone.js    ← bundles everything into standalone.html
│   └── download-models.js     ← fetches pre-trained models
└── assets/
    ├── models/                ← downloaded models (gitignored)
    └── web/lib/
        ├── numjs.js           ← numerical computing library
        └── sketch_rnn.js      ← Sketch RNN inference engine
```

## Building APK

To run as a native Android app (tested on Galaxy Tab S9 FE+):

```bash
mkdir sketch-guide-apk && cd sketch-guide-apk
npm init -y
npm install @capacitor/core @capacitor/cli @capacitor/android
npx cap init "SketchGuide" "com.sketchguide.app" --web-dir www
mkdir www && cp ../sketch-guide/standalone.html www/index.html
npx cap add android
npx cap sync
cd android && ./gradlew assembleDebug
```

APK output: `android/app/build/outputs/apk/debug/app-debug.apk`

> Requires Node.js >= 22 and Android SDK.

## Supported Categories

`pig`, `cat`, `dog`, `bird`, `flower`

More categories can be added by editing the `categories` array in `build-standalone.js` and re-downloading models. Sketch RNN supports [100+ categories](https://github.com/magenta/magenta-demos/blob/master/sketch-rnn-js/predict.js).

## Credits

- [Sketch RNN](https://arxiv.org/abs/1704.03477) by David Ha and Douglas Eck (Google Brain)
- [magenta-demos](https://github.com/magenta/magenta-demos) for the JS inference library
- [Quick Draw dataset](https://quickdraw.withgoogle.com/data) by Google Creative Lab

## License

MIT
