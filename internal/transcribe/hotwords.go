package transcribe

import (
	"context"
	"fmt"
	"os"
	"strings"
	"unicode"

	"github.com/joe/dndscribe-go/internal/db"
)

// commonWords is a set of English words that should never be auto-added as
// hotwords from transcription corrections. Ported from dndscribe-ts.
var commonWords = map[string]bool{
	"i": true, "me": true, "my": true, "myself": true, "we": true, "our": true,
	"ours": true, "ourselves": true, "you": true, "your": true, "yours": true,
	"yourself": true, "yourselves": true, "he": true, "him": true, "his": true,
	"himself": true, "she": true, "her": true, "hers": true, "herself": true,
	"it": true, "its": true, "itself": true, "they": true, "them": true,
	"their": true, "theirs": true, "themselves": true, "what": true, "which": true,
	"who": true, "whom": true, "this": true, "that": true, "these": true,
	"those": true, "am": true, "is": true, "are": true, "was": true, "were": true,
	"be": true, "been": true, "being": true, "have": true, "has": true, "had": true,
	"having": true, "do": true, "does": true, "did": true, "doing": true,
	"a": true, "an": true, "the": true, "and": true, "but": true, "if": true,
	"or": true, "because": true, "as": true, "until": true, "while": true,
	"of": true, "at": true, "by": true, "for": true, "with": true, "about": true,
	"against": true, "between": true, "through": true, "during": true,
	"before": true, "after": true, "above": true, "below": true, "to": true,
	"from": true, "up": true, "down": true, "in": true, "out": true, "on": true,
	"off": true, "over": true, "under": true, "again": true, "further": true,
	"then": true, "once": true, "here": true, "there": true, "when": true,
	"where": true, "why": true, "how": true, "all": true, "both": true,
	"each": true, "few": true, "more": true, "most": true, "other": true,
	"some": true, "such": true, "no": true, "nor": true, "not": true,
	"only": true, "own": true, "same": true, "so": true, "than": true,
	"too": true, "very": true, "can": true, "will": true, "just": true,
	"don": true, "should": true, "now": true, "also": true, "back": true,
	"could": true, "would": true, "into": true, "well": true, "like": true,
	"right": true, "yeah": true, "okay": true, "oh": true, "um": true, "uh": true,
	"yes": true, "got": true, "get": true, "go": true, "going": true, "gone": true,
	"come": true, "came": true, "know": true, "think": true, "want": true,
	"look": true, "make": true, "say": true, "said": true, "tell": true,
	"told": true, "take": true, "took": true, "see": true, "saw": true,
	"give": true, "gave": true, "put": true, "let": true, "keep": true,
	"kept": true, "still": true, "try": true, "tried": true, "thing": true,
	"things": true, "something": true, "nothing": true, "everything": true,
	"anything": true, "someone": true, "everyone": true, "anyone": true,
	"people": true, "man": true, "woman": true, "one": true, "two": true,
	"three": true, "four": true, "five": true, "first": true, "last": true,
	"new": true, "old": true, "good": true, "bad": true, "great": true,
	"little": true, "big": true, "long": true, "way": true, "day": true,
	"time": true, "part": true, "much": true, "many": true, "really": true,
	"actually": true,
}

// SyncHotwordsFile writes the union of all campaigns' hotwords to a text file.
// WHY: The whisper engine uses this for vocabulary biasing. We write all
// campaigns because the engine is global -- it doesn't know about campaigns.
func SyncHotwordsFile(filePath string) error {
	ctx := context.Background()
	words, err := db.GetAllHotwords(ctx)
	if err != nil {
		return fmt.Errorf("get all hotwords: %w", err)
	}

	content := ""
	if len(words) > 0 {
		content = strings.Join(words, "\n") + "\n"
	}

	if err := os.WriteFile(filePath, []byte(content), 0o644); err != nil {
		return fmt.Errorf("write hotwords file: %w", err)
	}
	return nil
}

// DetectNewHotwords compares original and edited text to find new proper nouns.
// Returns candidate hotwords: capitalized words in editedText that weren't in
// originalText and aren't common English words.
func DetectNewHotwords(originalText, editedText string) []string {
	strip := func(s string) string {
		var b strings.Builder
		for _, r := range s {
			if unicode.IsPunct(r) {
				continue
			}
			b.WriteRune(r)
		}
		return b.String()
	}

	// Build set of lowercase original words.
	origWords := make(map[string]bool)
	for _, raw := range strings.Fields(originalText) {
		origWords[strings.ToLower(strip(raw))] = true
	}

	var candidates []string
	seen := make(map[string]bool)

	for _, raw := range strings.Fields(editedText) {
		word := strip(raw)
		if len(word) < 2 {
			continue
		}
		// Must start with uppercase letter.
		first := rune(word[0])
		if !unicode.IsUpper(first) {
			continue
		}
		lower := strings.ToLower(word)
		// Skip if it was already in the original text.
		if origWords[lower] {
			continue
		}
		// Skip common English words.
		if commonWords[lower] {
			continue
		}
		// Deduplicate.
		if seen[lower] {
			continue
		}
		seen[lower] = true
		candidates = append(candidates, word)
	}

	return candidates
}

// BuildInitialPrompt creates the whisper initial prompt from hotwords and
// previous context. The prompt biases whisper toward recognizing these words.
func BuildInitialPrompt(hotwords []string, previousContext string) string {
	var parts []string

	if len(hotwords) > 0 {
		parts = append(parts, "Dungeons and Dragons RPG session. Names and terms: "+strings.Join(hotwords, ", ")+".")
	} else {
		parts = append(parts, "Dungeons and Dragons RPG session with fantasy names and places.")
	}

	if previousContext != "" {
		parts = append(parts, previousContext)
	}

	return strings.Join(parts, " ")
}
