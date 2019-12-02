import { ipcRenderer } from "electron";

process.once("loaded", () => {
  window.addEventListener("message", event => {
    const message = event.data;

    if (message.type == "ipcRenderer_SEND_DATA") {
      ipcRenderer.send("SEND_DATA", message.data);
    }
  });
});

// window.addEventListener("DOMContentLoaded", () => {
//   const replaceText = (selector: string, text: string) => {
//     const element = document.getElementById(selector);
//     if (element) {
//       element.innerText = text;
//     }
//   };
//
//   for (const type of ["chrome", "node", "electron"]) {
//     replaceText(`${type}-version`, (process.versions as any)[type]);
//   }
// });
