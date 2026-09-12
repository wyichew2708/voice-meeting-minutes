#!/bin/zsh
# Twelve macOS TTS voices, eight meeting-flavoured utterances each, 16 kHz mono
# 16-bit WAV. The speech test signal for sim/campp_reference.py: real speech
# through a real speaker model, which the synthetic geometry in speakers.py
# is not. macOS only (`say`). Not committed — regenerates in under a minute.
#
#   ./sim/make_tts_corpus.sh [out_dir]      default sim/corpus
set -e
OUT=${1:-"$(dirname "$0")/corpus"}
VOICES=(Samantha Daniel Karen Moira Tessa Fred Rishi Tara Aman Kathy Ralph Albert)
TEXTS=(
  "Yes, agreed."
  "Can we come back to that after the break?"
  "The claims backlog is sitting at about three hundred and forty cases this week."
  "I think Aisha should own the medical reports item, she has the context from last quarter."
  "Let's hire two temps for six weeks and review the numbers again at the end of October."
  "Right."
  "Before we move on, has anyone actually spoken to the underwriting team about the new form, or are we assuming they have seen it?"
  "So the decision is we go with the phased rollout, starting with the Singapore office, and we revisit the budget once the first month of data is in."
)
command -v say >/dev/null || { echo "needs macOS 'say'" >&2; exit 1; }
for v in $VOICES; do
  mkdir -p "$OUT/$v"
  i=0
  for t in $TEXTS; do
    i=$((i+1))
    [ -s "$OUT/$v/$i.wav" ] && continue
    say -v "$v" -o "$OUT/$v/$i.wav" --data-format=LEI16@16000 "$t" 2>/dev/null \
      || { say -v "$v" -o "$OUT/$v/$i.aiff" "$t" && afconvert -f WAVE -d LEI16@16000 -c 1 "$OUT/$v/$i.aiff" "$OUT/$v/$i.wav" && rm "$OUT/$v/$i.aiff"; }
  done
done
echo "$(find "$OUT" -name '*.wav' | wc -l | tr -d ' ') wavs in $OUT"
