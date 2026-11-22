package dj

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCleanRelPath(t *testing.T) {
	ok := map[string]string{
		"":                    "",
		"/":                   "",
		"Combat":              "Combat",
		"Combat/Boss":         "Combat/Boss",
		`Combat\Boss\a.mp3`:   "Combat/Boss/a.mp3",
		"/Combat/Boss/":       "Combat/Boss",
		"Vol. 2/01 Intro.mp3": "Vol. 2/01 Intro.mp3",
		"console.mp3":         "console.mp3",
	}
	for in, want := range ok {
		got, err := CleanRelPath(in)
		if err != nil || got != want {
			t.Errorf("CleanRelPath(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	bad := []string{
		"..", "../x.mp3", "a/../../x.mp3", "a/./b", "a//b",
		"C:/x.mp3", "a:b", "what?.mp3", "Boss.", "Boss /a.mp3", "a/ lead",
		"CON", "nul.mp3", "a/LPT1/x.mp3", "tab\there",
	}
	for _, in := range bad {
		if got, err := CleanRelPath(in); err == nil {
			t.Errorf("CleanRelPath(%q) = %q, want an error", in, got)
		}
	}
}

func TestAbsStaysInside(t *testing.T) {
	dir := t.TempDir()
	l := NewLibrary("ffmpeg", dir)
	full, err := l.abs("Combat/Boss")
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(dir, "Combat", "Boss"); full != want {
		t.Fatalf("abs = %q, want %q", full, want)
	}
	// abs is the second line of defence; it must refuse even if a raw,
	// uncleaned path reaches it.
	if _, err := l.abs("../outside"); err == nil {
		t.Fatal("abs accepted a path outside the music dir")
	}
}

func TestFoldersAndInFolder(t *testing.T) {
	dir := t.TempDir()
	for _, d := range []string{"Combat/Boss", "Tavern", "empty", ".hidden/x"} {
		if err := os.MkdirAll(filepath.Join(dir, filepath.FromSlash(d)), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	got, err := NewLibrary("ffmpeg", dir).Folders()
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"Combat", "Combat/Boss", "empty", "Tavern"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Folders = %v, want %v", got, want)
	}

	cases := []struct {
		track, folder string
		want          bool
	}{
		{"Combat/a.mp3", "Combat", true},
		{"Combat/Boss/b.mp3", "Combat", true},
		{"Combat/Boss/b.mp3", "Combat/Boss", true},
		{"Combat/a.mp3", "Combat/Boss", false},
		{"Combative/a.mp3", "Combat", false}, // prefix of a name, not a folder
		{"loose.mp3", "", true},
	}
	for _, c := range cases {
		if got := inFolder(c.track, c.folder); got != c.want {
			t.Errorf("inFolder(%q, %q) = %v, want %v", c.track, c.folder, got, c.want)
		}
	}
}
