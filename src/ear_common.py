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
     "Bender" alone waits up to FOLLOW_UP_S for the next utterance. A request that stops
     mid-sentence (`cut`, turn_common) is held HOLD_S for the rest, and heard again as one. Text that Whisper is
     known to invent from noise is dropped. While the robot itself speaks, the ear is MUTED
     (it would otherwise hear and answer itself), plus UNMUTE_GUARD_S after — except for its
     name: muted with `barge`, it still listens, and "Bender" alone or "Bender, …" interrupts
     the answer (a "barge" event). The page turns `barge` off for an answer that says "Bender"
     itself, or the robot would cut itself off.

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
# A request that stops mid-sentence ("Bender, acordate de que yo llevo…" — a pause to think closes
# the utterance) is held this long after it ended for the rest to arrive; then it goes as is.
HOLD_S = 2.0
HOLD_GAP_S = 0.3                  # silence put between the two halves when they are heard again

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


def recording(transcribe, folder, clock=time.time):
    """`transcribe`, keeping each utterance it is given as a WAV in `folder` and what it heard
    in `folder`/heard.jsonl ({"wav", "at", "seconds", "text"}). To measure the ear on real
    voices — which model hears them best, and whether END_SILENCE_MS cuts a sentence in two
    ("Acordate de que yo llevo" came in alone, 2026-10-28) — off unless EAR_RECORD_DIR is set:
    it records the room."""
    import json
    import os
    os.makedirs(folder, mode=0o700, exist_ok=True)

    def heard(wav):
        at = clock()
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(at)) + f"-{int(at * 1000) % 1000:03d}.wav"
        with open(os.path.join(folder, name), "wb") as f:
            f.write(wav)
        text = transcribe(wav)
        with open(os.path.join(folder, "heard.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"wav": name, "at": round(at, 3),
                                "seconds": round((len(wav) - 44) / (2 * RATE), 2),
                                "text": text or ""}, ensure_ascii=False) + "\n")
        return text
    return heard


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


_PUNCT = " ,.;:!¡?¿-…"


def _after(text, end):
    """What follows a wake word, without the punctuation and any repeat of the name: "Bender,
    Bender" asked "Bender" as a question (2026-10-28)."""
    rest = text[end:].strip(_PUNCT)
    while (m := WAKE.match(rest)):
        rest = rest[m.end():].strip(_PUNCT)
    return rest.strip()


def wake_command(text):
    """(woke, command) for an utterance: is "Bender" in it, and what comes after it."""
    m = WAKE.search(text or "")
    if not m:
        return False, ""
    return True, _after(text, m.end())


def last_wake_command(text):
    """Like wake_command, after the LAST "Bender": heard while the robot talks, the utterance
    starts with the robot's own words ("…la mesa es marrón. Bender, ¿y la silla?")."""
    found = list(WAKE.finditer(text or ""))
    if not found:
        return False, ""
    return True, _after(text, found[-1].end())


class Ear:
    """One listening session. feed() takes PCM and returns events for the browser:

      {"type": "speech"}                   someone started talking (for the UI)
      {"type": "heard", "text": ...}       what one utterance said (also when ignored)
      {"type": "wake"}                     "Bender" alone: say "¿Sí?", the next one is the command
      {"type": "command", "text": ...}     the request to answer
      {"type": "barge", "text": ...}       "Bender" while the robot talks: stop talking, then
                                           answer `text`, or with "" say "¿Sí?" as for a wake
    """

    def __init__(self, transcribe, clock=time.monotonic, cut=lambda text: False):
        self.transcribe = transcribe          # wav bytes -> text
        self.clock = clock
        self.cut = cut                        # text -> stopped mid-sentence? (turn_common)
        self.seg = Segmenter()
        self.muted = False
        self.barge = False
        self.unmuted_at = 0.0
        self.awaiting_until = 0.0
        self.held = None                      # a request cut off mid-sentence: {pcm, until, ...}

    def mute(self, on, barge=False):
        """`on` while the robot talks; `barge`: still listen for its name, to interrupt."""
        self.barge = bool(on and barge)
        if on and not self.muted:
            self.muted = True
            self.seg.reset()
        elif not on and self.muted:
            self.muted = False
            self.unmuted_at = self.clock()

    def feed(self, pcm):
        if self.muted and not self.barge:
            return []
        if not self.muted and self.clock() - self.unmuted_at < UNMUTE_GUARD_S:
            return []
        events = []
        for utterance in self.seg.push(pcm):
            events += self.barge_in(utterance) if self.muted else self.hear(utterance)
        if self.held and self.clock() > self.held["until"] and self.seg.utterance is None:
            # Nothing more came: the request goes as it was said.
            events.append({"type": "command", "text": self.held["request"]})
            self.held = None
        if self.seg.started:
            self.seg.started = False
            if not self.muted:                # not the robot's own voice as "someone talks"
                events.insert(0, {"type": "speech"})
        return events

    def barge_in(self, pcm):
        """An utterance heard while the robot talks: only its name counts (the rest is mostly
        the robot's own voice)."""
        text = (self.transcribe(pcm_to_wav(pcm)) or "").strip()
        if not text or HALLUCINATIONS.search(text):
            return []
        woke, command = last_wake_command(text)
        if not woke:
            return []
        self.awaiting_until = 0.0 if command else self.clock() + FOLLOW_UP_S
        return [{"type": "heard", "text": text}, {"type": "barge", "text": command}]

    def hear(self, pcm):
        # When the person STARTED talking: the follow-up window is for starting the question,
        # not for finishing it. Checked at the end, a long question after "¿Sí?" (the ack, the
        # pause, ~4 s of speech, the closing silence, the transcription) fell outside the 8 s
        # and was heard but not answered (2026-10-28).
        began = self.clock() - len(pcm) / (2 * RATE)
        held, self.held = self.held, None
        late = []
        if held and began < held["until"]:
            # The rest of a request cut off mid-sentence: heard again as one, so Whisper has
            # the whole sentence ("…yo llevo" + "la notebook a la reunión").
            pcm = held["pcm"] + bytes(int(HOLD_GAP_S * RATE) * 2) + pcm
        elif held:                            # started after the wait: that one goes as it was
            late, held = [{"type": "command", "text": held["request"]}], None
        text = (self.transcribe(pcm_to_wav(pcm)) or "").strip()
        if not text or HALLUCINATIONS.search(text):
            return [{"type": "command", "text": held["request"]}] if held else late
        events = [*late, {"type": "heard", "text": text}]
        woke, command = wake_command(text)
        request = ""
        if woke and command:
            request = command
        elif woke:
            self.awaiting_until = self.clock() + FOLLOW_UP_S
            events.append({"type": "wake"})
        elif began < self.awaiting_until or held:
            request = ACK_ECHO.sub("", text).strip()   # a bare "sí" is the robot's own ack
        if request:
            self.awaiting_until = 0.0
            # Asked on the words as heard: the request has lost its closing "?" (_after).
            if not held and self.cut(text):
                # Once: a second pause gets what was said so far, not another wait.
                self.held = {"pcm": pcm, "request": request, "until": self.clock() + HOLD_S}
            else:
                events.append({"type": "command", "text": request})
        return events
