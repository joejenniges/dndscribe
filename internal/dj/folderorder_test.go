package dj

import (
	"reflect"
	"testing"
)

func TestOrderFolders(t *testing.T) {
	paths := []string{"b", "a", "c", "a/y", "a/x", "a/x/deep", "C2"}
	pos := map[string]int{"c": 0, "a": 1, "a/y": 0}
	got := orderFolders(paths, pos)
	want := []string{"c", "a", "a/y", "a/x", "a/x/deep", "b", "C2"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v\nwant %v", got, want)
	}
}
