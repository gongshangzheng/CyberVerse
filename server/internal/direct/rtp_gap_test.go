package direct

import (
	"testing"
	"time"
)

func TestCappedRTPGap(t *testing.T) {
	t.Parallel()
	if got := cappedRTPGap(10 * time.Second); got != maxRTPTimestampGap {
		t.Fatalf("expected cap %v, got %v", maxRTPTimestampGap, got)
	}
	if got := cappedRTPGap(500 * time.Millisecond); got != 500*time.Millisecond {
		t.Fatalf("expected 500ms, got %v", got)
	}
}

func TestRTPGapThresholdUsesFrameDuration(t *testing.T) {
	t.Parallel()
	frameDur := time.Second / 25
	if got := rtpGapThreshold(frameDur); got != 2*frameDur {
		t.Fatalf("expected %v, got %v", 2*frameDur, got)
	}
}
