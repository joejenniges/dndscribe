package discordgo

import (
	"bytes"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"testing"
)

// The send path (EncryptFrame) was never exercised before the bot started
// playing music. These tests check it against the receive path, which is
// known to work with real Discord clients: a frame the bot encrypts must
// decrypt with the key a receiver derives for the bot's user ID, across key
// ratchet generations, and its truncated tag must be a real AES-GCM tag
// (receivers here skip tag checks, but Discord clients don't).

func newTestDAVE(t *testing.T, userID string) *DAVESession {
	t.Helper()
	d := NewDAVESession(userID)
	d.exporterSecret = make([]byte, 32)
	if _, err := rand.Read(d.exporterSecret); err != nil {
		t.Fatal(err)
	}
	if err := d.DeriveSenderKey(); err != nil {
		t.Fatal(err)
	}
	return d
}

func TestDAVESendRoundTrip(t *testing.T) {
	const botID = "1234567890123456789"
	d := newTestDAVE(t, botID)
	rs, err := d.DeriveReceiverKey(botID)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		opus := bytes.Repeat([]byte{byte(i), 0xAB}, 40+i)
		frame, err := d.EncryptFrame(opus)
		if err != nil {
			t.Fatal(err)
		}
		got, err := DecryptFrame(rs, frame)
		if err != nil {
			t.Fatalf("frame %d: %v", i, err)
		}
		if !bytes.Equal(got, opus) {
			t.Fatalf("frame %d: round trip mismatch", i)
		}
	}
}

func TestDAVESendRatchetGeneration(t *testing.T) {
	const botID = "42"
	d := newTestDAVE(t, botID)
	rs, err := d.DeriveReceiverKey(botID)
	if err != nil {
		t.Fatal(err)
	}
	// Jump to just before generation 1 (nonce >> 24), then cross it.
	d.senderNonce = 1<<24 - 2
	for i := 0; i < 4; i++ {
		opus := []byte{0xFC, byte(i), 1, 2, 3, 4, 5, 6}
		frame, err := d.EncryptFrame(opus)
		if err != nil {
			t.Fatal(err)
		}
		got, err := DecryptFrame(rs, frame)
		if err != nil || !bytes.Equal(got, opus) {
			t.Fatalf("nonce %d: err=%v match=%v", d.senderNonce, err, bytes.Equal(got, opus))
		}
	}
	if d.currentGeneration != 1 || rs.currentGen != 1 {
		t.Fatalf("generations sender=%d receiver=%d, want 1", d.currentGeneration, rs.currentGen)
	}
}

func TestDAVESendTagIsTruncatedGCM(t *testing.T) {
	const botID = "777"
	d := newTestDAVE(t, botID)
	opus := []byte("not really opus but fine")
	frame, err := d.EncryptFrame(opus)
	if err != nil {
		t.Fatal(err)
	}

	// Rebuild what a receiver computes: key for generation 0 from the
	// exported base secret, full 16-byte GCM tag, compare the first 8 bytes.
	rs, err := d.DeriveReceiverKey(botID)
	if err != nil {
		t.Fatal(err)
	}
	key, err := hashRatchetGetKey(rs.baseSecret, 0)
	if err != nil {
		t.Fatal(err)
	}
	block, _ := aes.NewCipher(key)
	gcm, _ := cipher.NewGCM(block)
	sealed := gcm.Seal(nil, buildNonce(1), opus, nil)
	wantTag := sealed[len(opus) : len(opus)+daveTagSize]

	gotTag := frame[len(opus) : len(opus)+daveTagSize]
	if !bytes.Equal(gotTag, wantTag) {
		t.Fatalf("tag %x, want %x", gotTag, wantTag)
	}
	if !bytes.Equal(frame[len(frame)-2:], []byte{0xFA, 0xFA}) {
		t.Fatal("missing 0xFAFA magic marker")
	}
	if int(frame[len(frame)-3]) != len(frame)-len(opus) {
		t.Fatalf("supplemental size %d, want %d", frame[len(frame)-3], len(frame)-len(opus))
	}
}
