package transcribe

import (
	"strings"
	"unicode"
)

// Hallucination filter thresholds, tuned from real session data.
// See dndscribe-ts transcription.ts for the calibration notes.
const (
	hallucinationRMSThreshold  float32 = 600
	hallucinationConfidenceMin float32 = 0.50
	hallucinationMaxWords              = 6
	veryLowRMSThreshold        float32 = 300
	veryLowRMSMaxWords                 = 3
)

// hallucinationBlocklist contains phrases Whisper commonly hallucinates
// from silence or low-energy audio. Stored lowercase, punctuation stripped.
var hallucinationBlocklist = map[string]bool{
	"thank you":                  true,
	"thank you very much":        true,
	"thanks":                     true,
	"thank you so much":          true,
	"thanks for watching":        true,
	"thank you for watching":     true,
	"thanks for listening":       true,
	"thank you for listening":    true,
	"subscribe":                  true,
	"please subscribe":           true,
	"like and subscribe":         true,
	"bye":                        true,
	"bye bye":                    true,
	"goodbye":                    true,
	"see you next time":          true,
	"see you in the next video":  true,
	"you":                        true,
	"the end":                    true,
	"ill see you in the next one": true,
	"yes":   true,
	"yeah":  true,
	"okay":  true,
	"oh":    true,
	"hmm":   true,
	"huh":   true,
	"uh":    true,
	"um":    true,
	"mhm":  true,
	"ah":    true,
	"so":    true,
	"right": true,
	"all right":    true,
	"alright":      true,
	"blank_audio":   true,
	"blank audio":   true,
	"blankaudio":    true,
	"music":         true,
	"music playing": true,
	"musicplaying":  true,
}

// stripPunctuation removes common punctuation characters for blocklist comparison.
func stripPunctuation(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if unicode.IsPunct(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

// IsHallucination checks if transcription text should be filtered out.
// Uses a blocklist of known Whisper hallucination phrases plus a
// multi-signal heuristic based on RMS energy, confidence, and word count.
func IsHallucination(text string, rms float32, confidence float32) bool {
	trimmed := strings.TrimSpace(text)
	if len(trimmed) < 2 {
		return true
	}

	// Blocklist check: exact match, case-insensitive, punctuation stripped.
	normalized := strings.ToLower(stripPunctuation(trimmed))
	normalized = strings.TrimSpace(normalized)
	if hallucinationBlocklist[normalized] {
		return true
	}

	wordCount := len(strings.Fields(trimmed))

	// Heuristic 1: short + low RMS + low confidence.
	// Sensitive mics produce low-volume audio that Whisper fills with
	// repetitive phrases. RMS is the strongest discriminator.
	if wordCount <= hallucinationMaxWords && rms < hallucinationRMSThreshold && confidence < hallucinationConfidenceMin {
		return true
	}

	// Heuristic 2: very low RMS with short output (e.g. mic resonance).
	if rms < veryLowRMSThreshold && wordCount <= veryLowRMSMaxWords {
		return true
	}

	return false
}
