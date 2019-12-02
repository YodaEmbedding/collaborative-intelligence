require("source-map-support").install();

import { app, BrowserWindow, ipcMain } from "electron";
import * as net from "net";
import * as path from "path";
import * as split2 from "split2";
import * as stream from "stream";
import * as through2 from "through2";

// import { Client } from "./client";

let mainWindow: BrowserWindow;
// let client: Client;
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
  // socket.setEncoding("utf8");
  // socket.on("connect", () => { });

  socket
    // .pipe(split2(JSON.parse))
    // .pipe(es.split())  // event-stream
    .pipe(split2())
    .pipe(
      // through2({ objectMode: true, allowHalfOpen: false }, ...)
      through2(function(chunk, _encoding, next) {
        const json = JSON.parse(chunk) as string;
        // TODO intermediate processing? or push non-serialized data in IPC?
        this.push(JSON.stringify(json));
        // this.push(json);
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

  socket.connect(5680, "localhost");

  // client = new Client();
  // client.connect(5680, "localhost");
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
  // ipcMain.on("SEND_DATA", (event, data) => {
  //   client.send(data);
  // });
  // ipcMain.on("RECEIVE_DATA", (event, data) => {
  //
  // });
}

// TODO ensure correctly ordered event handler registration
// https://stackoverflow.com/questions/47597982/send-sync-message-from-ipcmain-to-ipcrenderer-electron
//
