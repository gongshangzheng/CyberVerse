package mediapeer

import (
	"testing"
	"time"
)

func TestEncodePCMToOpusSamplesPadsTrailingBytes(t *testing.T) {
	const sampleRate = 16000
	bytesPerFrame := (sampleRate / 50) * 2

	// 52800 bytes matches a 33-frame @20fps segment but is not a multiple of 640.
	pcm := make([]byte, 52800)
	for i := range pcm {
		if i%2 == 0 {
			pcm[i] = 0x10
		}
	}

	samples, err := EncodePCMToOpusSamples(pcm, sampleRate)
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	wantFrames := len(pcm) / bytesPerFrame
	if len(pcm)%bytesPerFrame != 0 {
		wantFrames++
	}
	if len(samples) != wantFrames {
		t.Fatalf("expected %d opus frames, got %d", wantFrames, len(samples))
	}
	var audioDur time.Duration
	for _, s := range samples {
		audioDur += s.Duration
	}
	if audioDur != time.Duration(wantFrames)*20*time.Millisecond {
		t.Fatalf("unexpected audio duration %v", audioDur)
	}
}
