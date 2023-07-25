const { loadTFLiteModel } = require('tfjs-tflite-node');
const tf = require('@tensorflow/tfjs');
const tfnode = require("@tensorflow/tfjs-node")
const { Image } = require('canvas');
const fetch = require('node-fetch');
const fs = require('fs');
let model;

const loadModel = async () => {
    console.log('Loading Model...');
    model = await loadTFLiteModel('/home/wenyang/Machine/MoodGGDesktopForOW/model/model.tflite');
    console.log('Model loaded!');
};

async function predict() {
  console.log('Processing image...');
  console.log('Model:', model);

  // const res = await fetch('https://www.google.com/favicon.ico');
  // const buffer = res && res.ok ? await res.buffer() : null;
  // console.log('buffer:', buffer);
  const logits = tf.tidy(() => {
    const buffer = fs.readFileSync("/home/wenyang/Machine/MoodGGDesktopForOW/hand.jpg");
    // const decode = tfnode.node.decodeImage(buffer, 3);
    const decode = tfnode.node.decodeImage(new Uint8Array(buffer), 3);
    // const expand = tf.expandDims(decode, 0);
    // const expanded = tf.expandDims(buffer, 0);
    // const divided = tf.div(expanded, tf.scalar(127));
    // const normalized = tf.sub(divided, tf.scalar(1));
    console.log('buffer:', decode);
    // const batched = decode.reshape([1, 224, 224, 3]);
    //调整图片大小
    // const imgResize = tf.image.resizeBilinear(decode, [224, 224]);
    //归一化
    const batched = decode.toFloat().sub(255 / 2).div(255 / 2).reshape([1, 224, 224, 3]);
    return model.predict(batched);
  });

//   const output = await logits.array();
//   console.log('Prediction:', output);
}

async function run() {
  try {
    await loadModel();
    predict();
  } catch (error) {
    console.error('Error:', error);
  }
}

run();
