// import { ipcRenderer } from "electron";
const electron = window.require("electron");
const ipcRenderer = electron.ipcRenderer;

const sendBtn = document.getElementById("send") as HTMLButtonElement;
const textInput = document.getElementById("text") as HTMLInputElement;
const frameNumberText = spanById("frameNumber");
const inferenceTimeText = spanById("inferenceTime");
const predictionsTable = tableById("predictions");
const dataPreviewText = spanById("dataPreview");
const monitorStatsRawText = spanById("monitorStatsRaw");
const tensorViewImage = imageById("tensor_view");

const blackPng = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAOAAAADgCAAAAAA/RjU9AAAASElEQVR4nO3BMQEAAADCoPVPbQo/oAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAICXAcTgAAG6EJuyAAAAAElFTkSuQmCC"

ipcRenderer.on("update", (_event, data: string) => {
  const d = JSON.parse(data);
  monitorStatsRawText.textContent =
    data.slice(0, 60) + (data.length > 60 ? "..." : "");
  frameNumberText.textContent = d["frameNumber"];
  inferenceTimeText.textContent = d["inferenceTime"];
  dataPreviewText.textContent =
    d["data"].slice(0, 60) + (d["data"].length > 60 ? "..." : "");
  tensorViewImage.src = d["data"] != "" ? d["data"] : blackPng
  updateTable(predictionsTable, predictionsFormatter(d["predictions"]));
});

function tableById(id: string): HTMLTableElement {
  return document.getElementById(id) as HTMLTableElement;
}

function spanById(id: string): HTMLSpanElement {
  return document.getElementById(id) as HTMLSpanElement;
}

function imageById(id: string): HTMLImageElement {
  return document.getElementById(id) as HTMLImageElement;
}

function updateTable(table: HTMLTableElement, xss: string[][]) {
  const tbodyPrev = table.getElementsByTagName("tbody")[0];
  const tbody = createTableBody(xss);
  table.removeChild(tbodyPrev);
  table.appendChild(tbody);
}

function createTableBody(xss: string[][]): HTMLTableSectionElement {
  const tbody = document.createElement("tbody");

  for (const xs of xss) {
    const tr = tbody.insertRow();
    for (const x of xs) {
      const td = tr.insertCell();
      const text = document.createTextNode(x);
      td.appendChild(text);
    }
  }

  return tbody;
}

function predictionsFormatter(predictions: any[][]): string[][] {
  predictions
    .forEach(xs => {
      xs[2] = Math.floor(100 * Number(xs[2])).toString() + "%";
    });
  return predictions;
}

sendBtn.addEventListener("click", () => {
  const value = (textInput as HTMLInputElement).value;
  if (value) {
    window.postMessage({ type: "ipcRenderer_SEND_DATA", data: value }, "*");
  }
});
