optionsSelector.ontransitionend = (e) => {
  if (e.propertyName === "height") {
    const index = [].findIndex.call(
      optionsSelector.children,
      (x) => x === e.target
    );
    canvasWrapper.style.outlineColor = [
      "#9400d3",
      "#4b0082",
      "#0000ff",
      "#00ff00",
      "#ffff00",
      "#ff7f00",
      "#ff0000",
    ][index];
    [
      () => (mirror.checked = false),
      () => (mirror.checked = true),
      () => (outputSelector.value = "mask"),
      () => (outputSelector.value = "skinonly"),
      () => (outputSelector.value = "input"),
      () => (maskDenoise.checked = false),
      () => (maskDenoise.checked = true),
    ][index]();
  }
};

const video = document.getElementById("video-input");

const runtimeInitialized = new Promise((resolve, reject) => {
  cv["onRuntimeInitialized"] = () => {
    opencvInitInfo.parentNode.removeChild(opencvInitInfo);
    resolve();
  };
});

const webcamInitialized = new Promise((resolve, reject) => {
  start.onclick = () => {
    start.parentNode.removeChild(start);
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then(function (stream) {
        video.srcObject = stream;
        video.play();
        setTimeout(resolve);
      })
      .catch(reject);
  };
});

let src, cap;

let method1_hsvSkinRangeLow, method1_hsvSkinRangeHigh;

let method2_hsvSkinRangeLow1, method2_hsvSkinRangeHigh1;
let method2_hsvSkinRangeLow2, method2_hsvSkinRangeHigh2;

Promise.all([runtimeInitialized, webcamInitialized]).then(() => {
  src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
  cap = new cv.VideoCapture(video);

  const scalarToMat = (scalar, type) =>
    new cv.Mat(video.height, video.width, type, [
      scalar[0] || 0,
      scalar[1] || 0,
      scalar[2] || 0,
      scalar[3] || 0,
    ]);

  method1_hsvSkinRangeLow = scalarToMat(
    [0, Math.round(0.23 * 255), 0, 0],
    cv.CV_8UC3
  );

  method1_hsvSkinRangeHigh = scalarToMat(
    [10, Math.round(0.8 * 255), 255, 255],
    cv.CV_8UC3
  );

  method2_hsvSkinRangeLow1 = scalarToMat(
    [0, Math.round(0.2 * 255), 0.4 * 255, 0],
    cv.CV_8UC3
  );

  method2_hsvSkinRangeHigh1 = scalarToMat(
    [0.1 * 180, 0.6 * 255, 255, 255],
    cv.CV_8UC3
  );

  method2_hsvSkinRangeLow2 = scalarToMat(
    [0.9 * 180, 0.2 * 255, 0.4 * 255, 0],
    cv.CV_8UC3
  );

  method2_hsvSkinRangeHigh2 = scalarToMat(
    [180, 0.6 * 255, 255, 255],
    cv.CV_8UC3
  );

  method3_hsvSkinRangeLow = scalarToMat(
    [0.65 * 180, 0.5 * 255, 0.4 * 255, 0],
    cv.CV_8UC3
  );

  method3_hsvSkinRangeHigh = scalarToMat(
    [0.75 * 180, 255, 255, 255],
    cv.CV_8UC3
  );

  setTimeout(processVideo);
});

const fps = 60;

let streaming = true;

function processVideo() {
  if (!streaming) {
    src.delete();
    return;
  }

  const begin = Date.now();

  cap.read(src);

  if (mirror.checked) {
    cv.flip(src, src, 1);
  }

  let input = src;

  const histogramEqEnabled = histogramEq.checked;

  if (histogramEqEnabled) {
    input = equalizeHist(src);
  }

  let skinMask;
  switch (methodSelector.value) {
    case "1":
      skinMask = getSkinMask_method1(input);
      break;

    case "2":
      skinMask = getSkinMask_method2(input);
      break;

    default:
      return;
  }

  if (maskDenoise.checked) {
    cv.medianBlur(skinMask, skinMask, 7);
  }

  const skinOnly = new cv.Mat();
  cv.bitwise_and(input, input, skinOnly, skinMask);

  const centroids = findCentroids(skinMask);

  let option = null;

  for (const { x, y, width, height } of centroids) {
    if (y > (video.height * 3) / 5) {
      const pos = x / (video.width - 80);
      option = Math.floor(pos * 7) + 1;
      if (option > 7) {
        option = 7;
      }
      break;
    }
  }

  if (option !== null) {
    optionsSelector
      .querySelectorAll(`:not(:nth-child(${option}))`)
      .forEach((el) => el.classList.remove("active"));
    const element = optionsSelector
      .querySelector(`:nth-child(${option})`)
      .classList.add("active");
  } else {
    optionsSelector
      .querySelectorAll("div")
      .forEach((el) => el.classList.remove("active"));
  }

  canvasWrapper
    .querySelectorAll(".centroid")
    .forEach((c) => canvasWrapper.removeChild(c));

  for (const { x, y, width, height } of centroids) {
    canvasWrapper.insertAdjacentHTML(
      "beforeend",
      `<div class="centroid" style="top: ${y}px; left: ${x}px; width: ${width}px; height: ${height}px;"></div>`
    );
  }

  switch (outputSelector.value) {
    case "mask":
      cv.imshow("canvas-output", skinMask);
      break;

    case "skinonly":
      cv.imshow("canvas-output", skinOnly);
      break;

    case "input":
      cv.imshow("canvas-output", input);
      break;
  }

  skinMask.delete();
  skinOnly.delete();

  if (histogramEqEnabled) {
    input.delete();
  }

  const elapsed = Date.now() - begin;
  setTimeout(processVideo, 1000 / fps - elapsed);
}

function findCentroids(mask) {
  const centroids = [];

  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(
    mask,
    contours,
    hierarchy,
    cv.RETR_LIST,
    cv.CHAIN_APPROX_NONE
  );
  hierarchy.delete();

  for (let i = 0; i < contours.size(); i++) {
    const contour = contours.get(i);
    const area = cv.contourArea(contour);

    if (area == 0) {
      contour.delete();
      continue;
    }

    const { x, y, width, height } = cv.boundingRect(contour);
    if (width > 80 && height > 80) {
      centroids.push({ x, y, width, height });
    }

    contour.delete();
  }

  contours.delete();

  return centroids;
}

function getSkinMask_method1(src) {
  const hsv = new cv.Mat();
  cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);

  const skinMask_hsv = new cv.Mat();
  cv.inRange(
    hsv,
    method1_hsvSkinRangeLow,
    method1_hsvSkinRangeHigh,
    skinMask_hsv
  );

  hsv.delete();

  return skinMask_hsv;
}

function getSkinMask_method2(src) {
  const hsv = new cv.Mat();
  cv.cvtColor(src, hsv, cv.COLOR_RGB2HSV);

  const skinMask_hsv1 = new cv.Mat();
  cv.inRange(
    hsv,
    method2_hsvSkinRangeLow1,
    method2_hsvSkinRangeHigh1,
    skinMask_hsv1
  );

  const skinMask_hsv2 = new cv.Mat();
  cv.inRange(
    hsv,
    method2_hsvSkinRangeLow2,
    method2_hsvSkinRangeHigh2,
    skinMask_hsv2
  );

  const skinMask_hsv = new cv.Mat();
  cv.bitwise_or(skinMask_hsv1, skinMask_hsv2, skinMask_hsv);

  hsv.delete();
  skinMask_hsv1.delete();
  skinMask_hsv2.delete();

  return skinMask_hsv;
}

function equalizeHist(rgba) {
  const yuv = new cv.Mat();
  cv.cvtColor(rgba, yuv, cv.COLOR_RGB2YUV);

  const yuvPlanes = new cv.MatVector();
  cv.split(yuv, yuvPlanes);
  const y = yuvPlanes.get(0);
  const yEq = new cv.Mat();

  cv.equalizeHist(y, yEq);

  yuvPlanes.set(0, yEq);
  const yuvEq = new cv.Mat();
  cv.merge(yuvPlanes, yuvEq);

  const rgbEq = new cv.Mat();
  cv.cvtColor(yuvEq, rgbEq, cv.COLOR_YUV2RGB);

  const rgbaEq = new cv.Mat();
  cv.cvtColor(rgbEq, rgbaEq, cv.COLOR_RGB2RGBA);

  rgbEq.delete();
  yuvEq.delete();
  yEq.delete();
  y.delete();
  yuvPlanes.delete();
  yuv.delete();

  return rgbaEq;
}
