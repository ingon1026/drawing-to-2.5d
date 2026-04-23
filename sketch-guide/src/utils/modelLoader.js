// Local model loader - all models are bundled with the app (offline)
// Each model is ~2.9MB (small version)

const modelMap = {
  pig: require("../../assets/models/pig.gen.json"),
  cat: require("../../assets/models/cat.gen.json"),
  dog: require("../../assets/models/dog.gen.json"),
  bird: require("../../assets/models/bird.gen.json"),
  flower: require("../../assets/models/flower.gen.json"),
  butterfly: require("../../assets/models/butterfly.gen.json"),
  elephant: require("../../assets/models/elephant.gen.json"),
  rabbit: require("../../assets/models/rabbit.gen.json"),
  frog: require("../../assets/models/frog.gen.json"),
  crab: require("../../assets/models/crab.gen.json"),
  owl: require("../../assets/models/owl.gen.json"),
  whale: require("../../assets/models/whale.gen.json"),
  bee: require("../../assets/models/bee.gen.json"),
  bear: require("../../assets/models/bear.gen.json"),
  lion: require("../../assets/models/lion.gen.json"),
  octopus: require("../../assets/models/octopus.gen.json"),
  spider: require("../../assets/models/spider.gen.json"),
  penguin: require("../../assets/models/penguin.gen.json"),
  snail: require("../../assets/models/snail.gen.json"),
  duck: require("../../assets/models/duck.gen.json"),
};

export function getModelData(category) {
  return modelMap[category] || null;
}

export function getAvailableCategories() {
  return Object.keys(modelMap);
}
