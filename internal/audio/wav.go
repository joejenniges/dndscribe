package audio

import (
	"encoding/binary"
	"fmt"
	"os"
)

// CreateWAVHeader builds a WAV file header for raw PCM data.
func CreateWAVHeader(dataSize, sampleRate, channels, bitsPerSample int) []byte {
	header := make([]byte, 44)
	byteRate := sampleRate * channels * (bitsPerSample / 8)
	blockAlign := channels * (bitsPerSample / 8)

	copy(header[0:4], "RIFF")
	binary.LittleEndian.PutUint32(header[4:8], uint32(36+dataSize))
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16) // fmt chunk size
	binary.LittleEndian.PutUint16(header[20:22], 1)  // PCM format
	binary.LittleEndian.PutUint16(header[22:24], uint16(channels))
	binary.LittleEndian.PutUint32(header[24:28], uint32(sampleRate))
	binary.LittleEndian.PutUint32(header[28:32], uint32(byteRate))
	binary.LittleEndian.PutUint16(header[32:34], uint16(blockAlign))
	binary.LittleEndian.PutUint16(header[34:36], uint16(bitsPerSample))
	copy(header[36:40], "data")
	binary.LittleEndian.PutUint32(header[40:44], uint32(dataSize))

	return header
}

// WriteWAV writes mono or stereo PCM data as a WAV file.
// samples should be raw s16le PCM bytes.
func WriteWAV(path string, samples []byte, sampleRate, channels, bitsPerSample int) error {
	header := CreateWAVHeader(len(samples), sampleRate, channels, bitsPerSample)

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create WAV file: %w", err)
	}
	defer f.Close()

	if _, err := f.Write(header); err != nil {
		return fmt.Errorf("write WAV header: %w", err)
	}
	if _, err := f.Write(samples); err != nil {
		return fmt.Errorf("write WAV data: %w", err)
	}
	return nil
}
