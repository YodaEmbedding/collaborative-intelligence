// import { ipcRenderer } from "electron";
const electron = window.require('electron');
const ipcRenderer = electron.ipcRenderer;

const sendBtn = document.getElementById("send");
const textInput = document.getElementById("text");
const frameNumberText = document.getElementById("frameNumber");
const inferenceTimeText = document.getElementById("inferenceTime");
const predictionsText = document.getElementById("predictions");
const dataPreviewText = document.getElementById("dataPreview");
const monitorStatsRawText = document.getElementById("monitorStatsRaw");

ipcRenderer.on("update", (_event, data) => {
  var d = JSON.parse(data);
  frameNumberText.textContent = d["frameNumber"];
  inferenceTimeText.textContent = d["inferenceTime"];
  predictionsText.textContent = d["predictions"];
  dataPreviewText.textContent = d["data"];
  monitorStatsRawText.textContent = data;
});

sendBtn.addEventListener("click", () => {
  const value = (textInput as HTMLInputElement).value;
  if (value) {
    window.postMessage({ type: "ipcRenderer_SEND_DATA", data: value }, "*");
  }
});
