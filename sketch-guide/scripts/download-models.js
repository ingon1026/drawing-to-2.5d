#!/usr/bin/env node
/**
 * Download pre-trained Sketch RNN models from Google Cloud Storage.
 * Large models (~12MB) are used when available, otherwise small (~2.9MB).
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

const modelsDir = path.join(__dirname, '../assets/models');
const categories = ['pig', 'cat', 'dog', 'bird', 'flower'];

const LARGE_BASE = 'https://storage.googleapis.com/quickdraw-models/sketchRNN/large_models/';
const SMALL_BASE = 'https://storage.googleapis.com/quickdraw-models/sketchRNN/models/';

if (!fs.existsSync(modelsDir)) fs.mkdirSync(modelsDir, { recursive: true });

function download(url, dest) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(dest);
    https.get(url, (res) => {
      if (res.statusCode !== 200) { reject(new Error(`HTTP ${res.statusCode}`)); return; }
      res.pipe(file);
      file.on('finish', () => { file.close(); resolve(); });
    }).on('error', reject);
  });
}

async function main() {
  for (const cat of categories) {
    const largeDest = path.join(modelsDir, `${cat}.large.gen.json`);
    const smallDest = path.join(modelsDir, `${cat}.gen.json`);

    // Download large model
    process.stdout.write(`  ${cat} (large)... `);
    try {
      await download(LARGE_BASE + cat + '.gen.json', largeDest);
      const size = (fs.statSync(largeDest).size / 1024 / 1024).toFixed(1);
      console.log(`${size} MB`);
    } catch (e) {
      console.log(`not available, trying small...`);
      await download(SMALL_BASE + cat + '.gen.json', smallDest);
      const size = (fs.statSync(smallDest).size / 1024 / 1024).toFixed(1);
      console.log(`  ${cat} (small): ${size} MB`);
    }
  }
  console.log('\nDone! Run "npm run build" to generate standalone.html');
}

console.log('Downloading Sketch RNN models...\n');
main().catch(console.error);
