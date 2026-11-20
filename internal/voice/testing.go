package voice

// EnableDirectOpusForTest switches the stream to treat every packet as raw
// unencrypted opus, bypassing all DAVE gating. Functional tests use this to
// synthesize Discord-like packet streams without a DAVE session — production
// packets are always DAVE-framed, so processPacket would otherwise drop
// everything the test sends (pre-DAVE packets are skipped, and post-DAVE
// non-DAVE packets are treated as corrupt). Not for production use.
func (s *UserStream) EnableDirectOpusForTest() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.directOpus = true
}
