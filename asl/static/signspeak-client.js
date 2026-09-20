/**
 * SignSpeakClient - ES module for browser-based real-time sign recognition.
 *
 * Captures webcam frames, scales and encodes to JPEG, prepends an 8-byte
 * timestamp header, and streams over WebSocket with client-side backpressure.
 */

export class SignSpeakClient {
  constructor({
    url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/recognize',
    fps = 15,
    maxWidth = 480,
    quality = 0.7,
  } = {}) {
    this.url = url;
    this.fps = fps;
    this.maxWidth = maxWidth;
    this.quality = quality;

    this.videoEl = null;
    this.stream = null;
    this.ws = null;
    this.offscreenCanvas = null;
    this.ctx = null;

    this.listeners = new Map();
    this.running = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectTimer = null;
    this.animFrameId = null;
    this.timerId = null;
    this.lastFrameTime = 0;
    this.frameIntervalMs = 1000 / fps;
  }

  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
    return this;
  }

  emit(event, data) {
    const list = this.listeners.get(event) || [];
    for (const cb of list) {
      try {
        cb(data);
      } catch (err) {
        console.error(`Error in event listener for ${event}:`, err);
      }
    }
  }

  async start(videoEl) {
    this.videoEl = videoEl;
    this.running = true;

    // 1. Initialize offscreen canvas
    this.offscreenCanvas = document.createElement('canvas');
    this.ctx = this.offscreenCanvas.getContext('2d', { willReadFrequently: true });

    // 2. Open camera (facingMode: user)
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();
    } catch (err) {
      this.emit('error', new Error('Failed to acquire camera: ' + err.message));
      throw err;
    }

    // 3. Connect WebSocket
    this._connect();
  }

  _connect() {
    if (!this.running) return;

    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this._startCaptureLoop();
    };

    this.ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'ready') {
          this.emit('ready', msg);
        } else if (msg.type === 'prediction') {
          this.emit('prediction', msg);
          if (msg.commit) {
            this.emit('commit', msg.commit, msg);
          }
        } else if (msg.type === 'reset') {
          this.emit('reset', msg);
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    this.ws.onerror = (err) => {
      this.emit('error', err);
    };

    this.ws.onclose = (event) => {
      this._stopCaptureLoop();
      this.emit('close', event);
      if (this.running && this.reconnectAttempts < this.maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 10000);
        this.reconnectAttempts++;
        this.reconnectTimer = setTimeout(() => this._connect(), delay);
      }
    };
  }

  _startCaptureLoop() {
    if ('requestVideoFrameCallback' in HTMLVideoElement.prototype) {
      const loop = (now, metadata) => {
        if (!this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        if (now - this.lastFrameTime >= this.frameIntervalMs) {
          this.lastFrameTime = now;
          this._sendFrame();
        }
        this.animFrameId = this.videoEl.requestVideoFrameCallback(loop);
      };
      this.animFrameId = this.videoEl.requestVideoFrameCallback(loop);
    } else {
      this.timerId = setInterval(() => {
        if (!this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
        this._sendFrame();
      }, this.frameIntervalMs);
    }
  }

  _stopCaptureLoop() {
    if (this.animFrameId && this.videoEl && 'cancelVideoFrameCallback' in this.videoEl) {
      this.videoEl.cancelVideoFrameCallback(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.timerId) {
      clearInterval(this.timerId);
      this.timerId = null;
    }
  }

  _sendFrame() {
    if (!this.videoEl || !this.videoEl.videoWidth || !this.videoEl.videoHeight) return;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    // Client-side backpressure: drop frame if socket buffer has > 256 KB pending
    if (this.ws.bufferedAmount > 256 * 1024) {
      return;
    }

    const vw = this.videoEl.videoWidth;
    const vh = this.videoEl.videoHeight;
    const scale = Math.min(1.0, this.maxWidth / vw);
    const targetW = Math.round(vw * scale);
    const targetH = Math.round(vh * scale);

    if (this.offscreenCanvas.width !== targetW || this.offscreenCanvas.height !== targetH) {
      this.offscreenCanvas.width = targetW;
      this.offscreenCanvas.height = targetH;
    }

    // Draw RAW unmirrored video frame
    this.ctx.drawImage(this.videoEl, 0, 0, targetW, targetH);

    const clientTsMs = performance.now();

    this.offscreenCanvas.toBlob(
      async (blob) => {
        if (!blob || !this.running || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;

        const jpegBuffer = await blob.arrayBuffer();
        // Construct binary packet: 8-byte big-endian float64 timestamp + JPEG payload
        const packet = new Uint8Array(8 + jpegBuffer.byteLength);
        const view = new DataView(packet.buffer);
        view.setFloat64(0, clientTsMs, false); // false = big-endian
        packet.set(new Uint8Array(jpegBuffer), 8);

        if (this.ws.bufferedAmount <= 256 * 1024) {
          this.ws.send(packet.buffer);
        }
      },
      'image/jpeg',
      this.quality
    );
  }

  reset() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ cmd: 'reset' }));
    }
  }

  stop() {
    this.running = false;
    this._stopCaptureLoop();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.videoEl) {
      this.videoEl.srcObject = null;
    }
  }
}
