// tslint:disable

import * as net from "net";
import * as readline from "readline";
import * as stream from "stream";
import * as through2 from "through2";

// TODO unused
export class Client extends stream.Duplex {
  private socket: net.Socket;
  // private rl: readline.Interface

  // constructor() {
  //   super()
  // }

  connect(port: number, addr: string) {
    this.socket = new net.Socket();

    // const rl = readline.createInterface({
    //   input: this.socket,
    //   output: process.stdout,
    //   // output: this,
    //   // output: this.outputStream, // TODO wrong... pipe?
    // });
    // this.pipe(process.stdout);

    // TODO how is read...

    // this.socket.on('data', this.onReceiveData);
    this.socket.connect(port, addr);
  }

  close() {
    // TODO rl.close();
  }

  recv() {
    // err... isn't this handled by on(data)?
  }

  // TODO write?
  send(buffer: Uint8Array | string) {
    console.log(`Send ${buffer}`);
    this.socket.write(buffer);
  }

  // private onReceiveData(data: Buffer) {
  //   console.log(`Receive ${data}`);
  //   // TODO append to internal buffer until a complete message
  //   // then, fire that message off to some other listener
  // }
}
