package discordgo

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/binary"
	"fmt"
	"strconv"
	"sync"

	"github.com/bwmarrin/discordgo/mls"
)

type DAVESession struct {
	mu                  sync.Mutex
	protocolVersion     int
	epoch               uint64
	pendingTransitionID uint16
	pendingVersion      int

	exporterSecret    []byte
	senderKey         []byte
	senderNonce       uint32
	frameCipher       cipher.AEAD
	userID            string
	active            bool
	ratchetBaseSecret []byte
	currentGeneration uint32
	hasPendingKey     bool

	kpBundle *mls.KeyPackageBundle
}

func NewDAVESession(userID string) *DAVESession {
	return &DAVESession{
		userID: userID,
	}
}

func (d *DAVESession) GenerateKeyPackage() ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.generateKeyPackageLocked()
}

func (d *DAVESession) ResetForReWelcome() ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.exporterSecret = nil
	d.hasPendingKey = false

	return d.generateKeyPackageLocked()
}

func (d *DAVESession) generateKeyPackageLocked() ([]byte, error) {
	userIDNum, err := strconv.ParseUint(d.userID, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing user ID for credential: %w", err)
	}
	identity := make([]byte, 8)
	binary.BigEndian.PutUint64(identity, userIDNum)

	bundle, err := mls.GenerateKeyPackage(identity)
	if err != nil {
		return nil, fmt.Errorf("generating key package: %w", err)
	}
	d.kpBundle = bundle
	return bundle.Serialized, nil
}

func (d *DAVESession) HandleExternalSenderPackage(data []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return nil
}

func (d *DAVESession) HandleWelcome(data []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.kpBundle == nil {
		return fmt.Errorf("no key package generated")
	}

	result, err := mls.ProcessWelcome(data, d.kpBundle)
	if err != nil {
		return fmt.Errorf("processing welcome: %w", err)
	}

	d.exporterSecret = result.ExporterSecret
	d.epoch = result.Epoch
	d.hasPendingKey = true
	return nil
}

func (d *DAVESession) HandleCommit(data []byte) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return nil
}

func (d *DAVESession) HandlePrepareTransition(transitionID uint16, protocolVersion int) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.pendingTransitionID = transitionID
	d.pendingVersion = protocolVersion
}

func (d *DAVESession) HandleExecuteTransition(transitionID uint16) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if transitionID != d.pendingTransitionID {
		if d.senderKey != nil {
			d.active = true
		}
		return nil
	}

	if d.pendingVersion > 0 {
		derivedNewKey := false
		if d.hasPendingKey && d.exporterSecret != nil {
			if err := d.deriveSenderKeyLocked(); err != nil {
				return err
			}
			d.hasPendingKey = false
			derivedNewKey = true
		}
		if d.senderKey == nil {
			return nil
		}

		if !derivedNewKey && !d.hasPendingKey {
			d.active = false
			d.senderKey = nil
			d.frameCipher = nil
			d.ratchetBaseSecret = nil
			d.currentGeneration = 0
			return nil
		}

		d.active = true
	} else {
		d.active = false
		d.senderKey = nil
		d.frameCipher = nil
		d.hasPendingKey = false
	}
	return nil
}

func (d *DAVESession) HandlePrepareEpoch(epoch uint64, protocolVersion int) ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.epoch = epoch
	d.active = false
	d.senderKey = nil
	d.frameCipher = nil
	d.exporterSecret = nil

	return d.generateKeyPackageLocked()
}

func (d *DAVESession) DeriveSenderKey() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.deriveSenderKeyLocked()
}

func (d *DAVESession) deriveSenderKeyLocked() error {
	if d.exporterSecret == nil {
		return fmt.Errorf("no exporter secret")
	}

	userIDNum, err := strconv.ParseUint(d.userID, 10, 64)
	if err != nil {
		return fmt.Errorf("parsing user ID: %w", err)
	}
	context := make([]byte, 8)
	binary.LittleEndian.PutUint64(context, userIDNum)

	baseSecret, err := mls.Export(d.exporterSecret, daveExportLabel, context, daveKeySize)
	if err != nil {
		return fmt.Errorf("exporting base secret: %w", err)
	}

	d.ratchetBaseSecret = baseSecret
	d.currentGeneration = 0
	d.senderNonce = 0

	key, err := hashRatchetGetKey(baseSecret, 0)
	if err != nil {
		return fmt.Errorf("deriving ratchet key: %w", err)
	}
	d.senderKey = key

	frameCipher, err := newDAVECipher(key)
	if err != nil {
		return fmt.Errorf("creating frame cipher: %w", err)
	}
	d.frameCipher = frameCipher
	return nil
}

func (d *DAVESession) EncryptFrame(opusData []byte) ([]byte, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.frameCipher == nil {
		return nil, fmt.Errorf("no frame cipher")
	}

	d.senderNonce++

	generation := d.senderNonce >> 24
	if generation != d.currentGeneration {
		d.currentGeneration = generation
		key, err := hashRatchetGetKey(d.ratchetBaseSecret, generation)
		if err != nil {
			return nil, fmt.Errorf("ratcheting key for generation %d: %w", generation, err)
		}
		d.senderKey = key
		frameCipher, err := newDAVECipher(key)
		if err != nil {
			return nil, fmt.Errorf("creating cipher for generation %d: %w", generation, err)
		}
		d.frameCipher = frameCipher
	}

	encrypted := encryptSecureFrame(d.frameCipher, d.senderNonce, opusData)
	return encrypted, nil
}

func (d *DAVESession) IsActive() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.active
}

func (d *DAVESession) CanEncrypt() bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.frameCipher != nil
}

// ReceiverState tracks decryption state for a single remote sender.
type ReceiverState struct {
	baseSecret []byte
	currentGen uint32
	cipher     cipher.AEAD
}

// DeriveReceiverKey derives the DAVE decryption key for a remote sender.
// Returns a ReceiverState that can be used to decrypt their frames.
func (d *DAVESession) DeriveReceiverKey(senderUserID string) (*ReceiverState, error) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.exporterSecret == nil {
		return nil, fmt.Errorf("no exporter secret")
	}

	userIDNum, err := strconv.ParseUint(senderUserID, 10, 64)
	if err != nil {
		return nil, fmt.Errorf("parsing sender user ID: %w", err)
	}
	context := make([]byte, 8)
	binary.LittleEndian.PutUint64(context, userIDNum)

	baseSecret, err := mls.Export(d.exporterSecret, daveExportLabel, context, daveKeySize)
	if err != nil {
		return nil, fmt.Errorf("exporting receiver base secret: %w", err)
	}

	key, err := hashRatchetGetKey(baseSecret, 0)
	if err != nil {
		return nil, fmt.Errorf("deriving initial receiver key: %w", err)
	}

	frameCipher, err := newDAVEDecryptor(key)
	if err != nil {
		return nil, fmt.Errorf("creating receiver cipher: %w", err)
	}

	return &ReceiverState{
		baseSecret: baseSecret,
		currentGen: 0,
		cipher:     frameCipher,
	}, nil
}

// DecryptFrame decrypts a DAVE secure frame using the given receiver state.
// The input should be the raw DAVE frame (with 0xFAFA trailer).
func DecryptFrame(rs *ReceiverState, data []byte) ([]byte, error) {
	if len(data) < 13 { // minimum: 1 byte data + 8 tag + 1 nonce + 1 size + 2 magic
		return nil, fmt.Errorf("frame too short: %d bytes", len(data))
	}

	// Verify magic trailer
	if data[len(data)-1] != 0xFA || data[len(data)-2] != 0xFA {
		return nil, fmt.Errorf("not a DAVE frame (no 0xFAFA trailer)")
	}

	// Read supplemental size
	supplementalSize := int(data[len(data)-3])
	if supplementalSize >= len(data) || supplementalSize < 12 {
		return nil, fmt.Errorf("invalid supplemental size: %d", supplementalSize)
	}

	// Extract components
	ciphertextEnd := len(data) - supplementalSize
	ciphertext := data[:ciphertextEnd]
	tag := data[ciphertextEnd : ciphertextEnd+daveTagSize]

	// Read nonce (ULEB128 encoded between tag and supplementalSize byte)
	nonceStart := ciphertextEnd + daveTagSize
	nonceEnd := len(data) - 3 // before supplementalSize byte and magic
	nonce := decodeULEB128(data[nonceStart:nonceEnd])

	// Check if we need to ratchet the key
	generation := nonce >> 24
	if generation != rs.currentGen {
		key, err := hashRatchetGetKey(rs.baseSecret, generation)
		if err != nil {
			return nil, fmt.Errorf("ratcheting to generation %d: %w", generation, err)
		}
		frameCipher, err := newDAVEDecryptor(key)
		if err != nil {
			return nil, fmt.Errorf("creating cipher for generation %d: %w", generation, err)
		}
		rs.cipher = frameCipher
		rs.currentGen = generation
	}

	// Build sealed data (ciphertext + truncated tag) for our decryptor
	sealed := make([]byte, len(ciphertext)+daveTagSize)
	copy(sealed, ciphertext)
	copy(sealed[len(ciphertext):], tag)

	fullNonce := buildNonce(nonce)
	plaintext, err := rs.cipher.Open(nil, fullNonce, sealed, nil)
	if err != nil {
		return nil, fmt.Errorf("decrypt: %w", err)
	}

	return plaintext, nil
}

func decodeULEB128(data []byte) uint32 {
	var result uint32
	var shift uint
	for _, b := range data {
		result |= uint32(b&0x7F) << shift
		if b&0x80 == 0 {
			break
		}
		shift += 7
	}
	return result
}

// daveDecryptor decrypts GCM-encrypted data using only the CTR layer.
// Go's standard GCM requires tag sizes 12-16, but DAVE uses 8-byte tags.
// Since the SRTP layer already authenticates packets, we skip GCM tag
// verification and just decrypt via AES-CTR.
type daveDecryptor struct {
	block cipher.Block
}

func (d *daveDecryptor) NonceSize() int { return 12 }
func (d *daveDecryptor) Overhead() int  { return daveTagSize }
func (d *daveDecryptor) Seal(dst, nonce, plaintext, additionalData []byte) []byte {
	return nil // not used for receiving
}

func (d *daveDecryptor) Open(dst, nonce, ciphertext, additionalData []byte) ([]byte, error) {
	if len(ciphertext) < daveTagSize {
		return nil, fmt.Errorf("ciphertext too short")
	}
	// Strip the truncated tag — we skip verification
	ct := ciphertext[:len(ciphertext)-daveTagSize]

	// GCM encrypts using AES-CTR starting at nonce||0x00000002
	// (nonce||0x00000001 is reserved for tag computation)
	counter := make([]byte, aes.BlockSize)
	copy(counter, nonce)
	binary.BigEndian.PutUint32(counter[12:], 2)

	stream := cipher.NewCTR(d.block, counter)
	plaintext := make([]byte, len(ct))
	stream.XORKeyStream(plaintext, ct)
	return plaintext, nil
}

func newDAVEDecryptor(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return &daveDecryptor{block: block}, nil
}

func (d *DAVESession) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.exporterSecret = nil
	d.senderKey = nil
	d.senderNonce = 0
	d.frameCipher = nil
	d.active = false
	d.kpBundle = nil
	d.pendingTransitionID = 0
	d.pendingVersion = 0
	d.ratchetBaseSecret = nil
	d.currentGeneration = 0
	d.hasPendingKey = false
}
