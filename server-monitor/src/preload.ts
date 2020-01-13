import { ipcRenderer } from "electron";

process.once("loaded", () => {
  window.addEventListener("message", event => {
    const message = event.data;

    if (message.type == "ipcRenderer_SEND_DATA") {
      ipcRenderer.send("SEND_DATA", message.data);
    }
  });
});
