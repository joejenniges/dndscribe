package db

import "unicode"

// NaturalLess orders strings case-insensitively with runs of digits
// compared as numbers: "Track 2" < "Track 10" < "track 11". Equal numbers
// with different zero padding ("01" vs "1") fall back to the shorter run
// first, and fully equal-ignoring-case strings to a byte comparison, so the
// order is total and stable across calls.
func NaturalLess(a, b string) bool {
	ra, rb := []rune(a), []rune(b)
	i, j := 0, 0
	for i < len(ra) && j < len(rb) {
		ca, cb := ra[i], rb[j]
		if isDigit(ca) && isDigit(cb) {
			si, sj := i, j
			for i < len(ra) && isDigit(ra[i]) {
				i++
			}
			for j < len(rb) && isDigit(rb[j]) {
				j++
			}
			na, nb := trimZeros(ra[si:i]), trimZeros(rb[sj:j])
			if len(na) != len(nb) {
				return len(na) < len(nb)
			}
			for k := range na {
				if na[k] != nb[k] {
					return na[k] < nb[k]
				}
			}
			if i-si != j-sj {
				return i-si < j-sj
			}
			continue
		}
		la, lb := sortKey(ca), sortKey(cb)
		if la != lb {
			return la < lb
		}
		i++
		j++
	}
	if len(ra)-i != len(rb)-j {
		return len(ra)-i < len(rb)-j
	}
	return a < b
}

// sortKey lowercases, and puts "/" first so a folder's own tracks sort
// together, before a sibling folder that shares its name as a prefix
// ("Boss/..." before "Boss Extra/...").
func sortKey(r rune) rune {
	if r == '/' {
		return 0
	}
	return unicode.ToLower(r)
}

func isDigit(r rune) bool { return r >= '0' && r <= '9' }

func trimZeros(r []rune) []rune {
	for len(r) > 1 && r[0] == '0' {
		r = r[1:]
	}
	return r
}
