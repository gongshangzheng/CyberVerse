package direct

import (
	"context"
	"testing"
	"time"

	"github.com/pion/webrtc/v4"
)

func TestStartNegotiationSendsOfferWithoutWaitingForICEGathering(t *testing.T) {
	t.Parallel()

	signals := make(chan map[string]any, 16)
	peer := NewDirectPeer(
		"session-test",
		func(_ string, msg map[string]any) {
			signals <- msg
		},
		[]webrtc.ICEServer{{URLs: []string{"stun:192.0.2.1:3478"}}},
		nil,
		nil,
	)
	if err := peer.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer func() {
		if err := peer.Disconnect(); err != nil {
			t.Fatalf("Disconnect: %v", err)
		}
	}()

	startedAt := time.Now()
	if err := peer.StartNegotiation(); err != nil {
		t.Fatalf("StartNegotiation: %v", err)
	}
	if elapsed := time.Since(startedAt); elapsed > time.Second {
		t.Fatalf("StartNegotiation took %v; expected offer before ICE gathering completes", elapsed)
	}

	deadline := time.After(time.Second)
	for {
		select {
		case msg := <-signals:
			if msg["type"] != "webrtc_offer" {
				continue
			}
			sdp, ok := msg["sdp"].(string)
			if !ok || sdp == "" {
				t.Fatalf("offer SDP missing: %#v", msg)
			}
			return
		case <-deadline:
			t.Fatal("timed out waiting for webrtc_offer")
		}
	}
}
