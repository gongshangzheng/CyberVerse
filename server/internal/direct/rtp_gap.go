package direct

import "time"

// maxRTPTimestampGap caps how much the first sample of a segment may advance
// the RTP clock after an idle period. Uncapped wall-clock gaps (e.g. 10s+
// between turns) inflate the media timeline and contribute to A/V drift.
const maxRTPTimestampGap = 2 * time.Second

func rtpGapThreshold(frameDur time.Duration) time.Duration {
	threshold := 2 * frameDur
	if threshold < 40*time.Millisecond {
		return 40 * time.Millisecond
	}
	return threshold
}

func cappedRTPGap(wall time.Duration) time.Duration {
	if wall <= 0 {
		return 0
	}
	if wall > maxRTPTimestampGap {
		return maxRTPTimestampGap
	}
	return wall
}
