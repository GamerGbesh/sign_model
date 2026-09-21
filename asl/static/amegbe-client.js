/**
 * AmegbeClient - Browser client for Amegbe: sign recognition + Twi speech synthesis.
 *
 * Extends SignSpeakClient with:
 * - AudioQueue: sequential playback via HTMLAudioElement/Web Audio API with queue bounding (max 3).
 * - Speech configuration (mode: 'word' | 'sentence' | 'off', voice selection).
 * - Real-time speech event handling and sentence controls (speakNow, clearSentence).
 */
import { SignSpeakClient } from './signspeak-client.js';

export class AudioQueue {
  constructor({ maxPending = 3, onAck = null } = {}) {
    this.maxPending = maxPending;
    this.onAck = onAck;
    this.queue = [];
    this.isPlaying = false;
    this.isMuted = false;
    this.volume = 1.0;
    this.currentAudio = null;
    this.audioContext = null;
  }

  initUserGesture() {
    if (!this.audioContext && (window.AudioContext || window.webkitAudioContext)) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx();
    }
    if (this.audioContext && this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }
  }

  enqueue(speechEvent) {
    if (this.isMuted) return;
    if (this.queue.length >= this.maxPending) {
      const dropped = this.queue.shift();
      console.warn('AudioQueue backpressure: dropped oldest audio clip', dropped);
    }
    this.queue.push(speechEvent);
    if (!this.isPlaying) {
      this._playNext();
    }
  }

  setMute(muted) {
    this.isMuted = Boolean(muted);
    if (this.isMuted) {
      this.stop();
    }
  }

  setVolume(vol) {
    this.volume = Math.max(0.0, Math.min(1.0, vol));
    if (this.currentAudio) {
      this.currentAudio.volume = this.volume;
    }
  }

  stop() {
    this.queue = [];
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio = null;
    }
    this.isPlaying = false;
  }

  _playNext() {
    if (this.queue.length === 0 || this.isMuted) {
      this.isPlaying = false;
      this.currentAudio = null;
      return;
    }
    this.isPlaying = true;
    const item = this.queue.shift();
    const audio = new Audio(item.audio_url);
    audio.volume = this.volume;
    this.currentAudio = audio;

    const cleanup = () => {
      if (this.onAck && item.audio_key) {
        this.onAck(item.audio_key);
      }
      this.currentAudio = null;
      this._playNext();
    };

    audio.onended = cleanup;
    audio.onerror = (e) => {
      console.error('Audio playback error:', item.audio_url, e);
      cleanup();
    };

    audio.play().catch((err) => {
      console.warn('Audio play blocked by browser autoplay policy:', err);
      cleanup();
    });
  }
}

export class AmegbeClient extends SignSpeakClient {
  constructor({
    url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws/recognize',
    fps = 15,
    maxWidth = 480,
    quality = 0.7,
    speechMode = 'word',
    voice = 'auto',
    maxPendingAudio = 3,
  } = {}) {
    super({ url, fps, maxWidth, quality });
    this.speechMode = speechMode;
    this.voice = voice;

    this.audioQueue = new AudioQueue({
      maxPending: maxPendingAudio,
      onAck: (key) => this.ackAudio(key),
    });

    // Handle speech events automatically
    this.on('speech', (event) => {
      if (this.speechMode !== 'off') {
        this.audioQueue.enqueue(event);
      }
    });

    this.on('ready', () => {
      this.configureSpeech({ speechMode: this.speechMode, voice: this.voice });
    });
  }

  async start(videoEl) {
    this.audioQueue.initUserGesture();
    return super.start(videoEl);
  }

  configureSpeech({ speechMode, voice }) {
    if (speechMode !== undefined) this.speechMode = speechMode;
    if (voice !== undefined) this.voice = voice;

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'config',
        speech_mode: this.speechMode,
        voice: this.voice,
      }));
    }
  }

  speakNow() {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'speak' }));
    }
  }

  clearSentence() {
    this.audioQueue.stop();
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'clear' }));
    }
  }

  ackAudio(key) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'ack_audio', key }));
    }
  }

  setMute(muted) {
    this.audioQueue.setMute(muted);
  }

  setVolume(vol) {
    this.audioQueue.setVolume(vol);
  }

  stop() {
    this.audioQueue.stop();
    super.stop();
  }
}
