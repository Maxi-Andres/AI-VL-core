#!/usr/bin/env python3
"""
ear_common.py — The robot's ear: a live microphone stream in, "Bender, ..." commands out.

The G1 streams its own microphone (16 kHz mono s16le, 160 ms packets) through its relay;
service.py's /ws/ear receives it from the backend. This module turns that stream into what
the voice assistant needs, Alexa-style:

  1. SEGMENTER — cuts the stream into utterances by energy. The threshold follows the noise:
     the G1's fans sit at an RMS of ~0.008 (never over 0.0092 in silence) while speech
     reaches 0.05-0.11 (measured 2026-10-19), so "speech" is a 30 ms frame over
     max(MIN_LEVEL, RATIO x noise floor). An utterance starts after START_FRAMES voiced
     frames (with PREROLL_MS of audio before it, so the first syllable is kept), ends after
     END_SILENCE_MS of silence or at MAX_S, and is dropped with under MIN_VOICED_S of voice.
  2. EAR — sends each utterance to speech-to-text (Whisper with the "Bender" hotword and its
     own VAD, asr_common) and looks for the wake word. "Bender, qué ves" is a command at once;
     "Bender" alone waits up to FOLLOW_UP_S for the next utterance. Text that Whisper is
     known to invent from noise is dropped. While the robot itself speaks, the ear is MUTED
     (it would otherwise hear and answer itself), plus UNMUTE_GUARD_S after.

Pure logic with an injected transcriber and clock, so tests/test_ear.py drives it with
made-up audio.
"""
import array
import math
import re
import struct
import time

RATE = 16000
FRAME = 480                       # 30 ms

MIN_LEVEL = 0.015                 # an RMS under this is never speech
RATIO = 2.5                       # ... nor anything under this many times the noise floor
START_FRAMES = 3                  # 90 ms of voice opens an utterance
END_SILENCE_MS = 800
PREROLL_MS = 300
MAX_S = 8.0
MIN_VOICED_S = 0.2                # less voice than this is a click, not a word ("sí" is ~0.25)
FOLLOW_UP_S = 8.0
UNMUTE_GUARD_S = 0.4

WAKE = re.compile(r"\bb[eé]nd[eé]r\b", re.IGNORECASE)
# The page answers a bare "Bender" with a spoken "¿Sí?" and does NOT mute the ear for it — muting
# made the ear deaf for the ~2 s the acknowledgement took to make and play, and a question asked
# right away was lost. So the ear hears its own "¿Sí?": alone it is ignored (the follow-up
# window stays open); run into the question ("Sí, ¿qué ves?") it is cut off the front.
ACK_ECHO = re.compile(r"^\W*s[ií]\b\W*", re.IGNORECASE)
# Whisper's well-known inventions on silence or noise (it learned from subtitled videos):
# anything with these phrases, and an utterance that is nothing but thanks — "Thank you, thank
# you, thank you" came out of a muddy stretch of the G1's mic on 2026-10-19, in English, with
# the language forced to Spanish, right while the ear waited for a command.
HALLUCINATIONS = re.compile(
    r"suscr[ií]bete|gracias por ver|subt[ií]tulos|amara\.org|thanks for watching"
    r"|^\W*(?:(?:thank you|thanks|gracias|you)\W*)+$",
    re.IGNORECASE)


def rms(pcm):
    """RMS of 16-bit samples, 0..1."""
    a = array.array("h", pcm[: len(pcm) // 2 * 2])
    if not a:
        return 0.0
    return math.sqrt(sum(x * x for x in a) / len(a)) / 32768


def pcm_to_wav(pcm, rate=RATE):
    """16-bit mono PCM bytes as a WAV file (what /transcribe takes)."""
    data = bytes(pcm[: len(pcm) // 2 * 2])
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


class Segmenter:
    """Cuts a PCM stream into utterances. push() returns the utterances it completed."""

    def __init__(self):
        self.buf = b""                        # bytes not yet framed
        self.floor = None                     # noise level, learned from non-speech frames
        self.preroll = []                     # the last frames before an utterance
        self.voiced = 0                       # consecutive voiced frames while idle
        self.utterance = None                 # list of frames while inside one
        self.silent = 0                       # consecutive silent frames inside one
        self.voiced_in = 0                    # voiced frames inside one
        self.started = False                  # an utterance opened since the last push()

    def reset(self):
        floor = self.floor
        self.__init__()
        self.floor = floor

    def threshold(self):
        return max(MIN_LEVEL, RATIO * (self.floor or MIN_LEVEL))

    def push(self, pcm):
        self.buf += pcm
        done = []
        nbytes = FRAME * 2
        while len(self.buf) >= nbytes:
            frame, self.buf = self.buf[:nbytes], self.buf[nbytes:]
            out = self._frame(frame)
            if out is not None:
                done.append(out)
        return done

    def _frame(self, frame):
        level = rms(frame)
        voice = level > self.threshold()
        if self.utterance is None:
            if not voice:
                self.floor = level if self.floor is None else 0.95 * self.floor + 0.05 * level
            self.preroll = [*self.preroll, frame][-(PREROLL_MS // 30):]
            self.voiced = self.voiced + 1 if voice else 0
            if self.voiced >= START_FRAMES:
                self.utterance, self.silent, self.voiced = list(self.preroll), 0, 0
                self.voiced_in = START_FRAMES
                self.preroll = []
                self.started = True
            return None
        self.utterance.append(frame)
        self.silent = 0 if voice else self.silent + 1
        self.voiced_in += 1 if voice else 0
        seconds = len(self.utterance) * FRAME / RATE
        if self.silent * 30 >= END_SILENCE_MS or seconds >= MAX_S:
            pcm = b"".join(self.utterance)
            self.utterance = None
            if self.voiced_in * FRAME / RATE >= MIN_VOICED_S:
                return pcm
        return None


def wake_command(text):
    """(woke, command) for an utterance: is "Bender" in it, and what comes after it."""
    m = WAKE.search(text or "")
    if not m:
        return False, ""
    return True, text[m.end():].strip(" ,.;:!¡?¿-…").strip()


class Ear:
    """One listening session. feed() takes PCM and returns events for the browser:

      {"type": "speech"}                   someone started talking (for the UI)
      {"type": "heard", "text": ...}       what one utterance said (also when ignored)
      {"type": "wake"}                     "Bender" alone: say "¿Sí?", the next one is the command
      {"type": "command", "text": ...}     the request to answer
    """

    def __init__(self, transcribe, clock=time.monotonic):
        self.transcribe = transcribe          # wav bytes -> text
        self.clock = clock
        self.seg = Segmenter()
        self.muted = False
        self.unmuted_at = 0.0
        self.awaiting_until = 0.0

    def mute(self, on):
        if on and not self.muted:
            self.muted = True
            self.seg.reset()
        elif not on and self.muted:
            self.muted = False
            self.unmuted_at = self.clock()

    def feed(self, pcm):
        if self.muted or self.clock() - self.unmuted_at < UNMUTE_GUARD_S:
            return []
        events = []
        for utterance in self.seg.push(pcm):
            events += self.hear(utterance)
        if self.seg.started:
            self.seg.started = False
            events.insert(0, {"type": "speech"})
        return events

    def hear(self, pcm):
        text = (self.transcribe(pcm_to_wav(pcm)) or "").strip()
        if not text or HALLUCINATIONS.search(text):
            return []
        events = [{"type": "heard", "text": text}]
        woke, command = wake_command(text)
        if woke and command:
            self.awaiting_until = 0.0
            events.append({"type": "command", "text": command})
        elif woke:
            self.awaiting_until = self.clock() + FOLLOW_UP_S
            events.append({"type": "wake"})
        elif self.clock() < self.awaiting_until:
            request = ACK_ECHO.sub("", text).strip()
            if request:                       # a bare "sí" is the robot's own ack: keep waiting
                self.awaiting_until = 0.0
                events.append({"type": "command", "text": request})
        return events
