require("source-map-support").install();

import { app, BrowserWindow, ipcMain } from "electron";
import * as net from "net";
import * as path from "path";
import * as split2 from "split2";
import * as stream from "stream";
import * as through2 from "through2";

const HOSTNAME: string = "localhost";
const PORT: number = 5680;

let mainWindow: BrowserWindow;
let socket: net.Socket;

app.on("activate", () => {
  if (mainWindow === null) {
    createWindow();
  }
});

app.on("ready", () => {
  createWindow();
  registerHandlers();

  socket = new net.Socket();

  socket
    .pipe(split2())
    .pipe(
      through2(function(chunk, _encoding, next) {
        const json = JSON.parse(chunk) as string;
        this.push(JSON.stringify(json));
        next();
      })
    )
    .pipe(
      new stream.Writable({
        write: function(chunk, _encoding, next) {
          mainWindow.webContents.send("update", chunk);
          console.log("update sent");
          next();
        }
      })
    );

  socket.connect(PORT, HOSTNAME);
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

process.on("uncaughtException", error => {
  console.log(error);
});

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      preload: path.join(__dirname, "preload.js")
    }
  });

  mainWindow.loadFile(path.join(__dirname, "../index.html"));

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function registerHandlers() {
}
