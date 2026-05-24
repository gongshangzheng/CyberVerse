package direct

import (
	"github.com/pion/interceptor"
	"github.com/pion/interceptor/pkg/cc"
	"github.com/pion/interceptor/pkg/gcc"
	"github.com/pion/webrtc/v4"
)

// WebRTCAPIConfig configures the shared webrtc.API with interceptors.
type WebRTCAPIConfig struct {
	InitialBitrate int // bps, default 1_000_000 (1 Mbps)
	MinBitrate     int // bps, default 100_000 (100 kbps)
	MaxBitrate     int // bps, default 10_000_000 (10 Mbps)
}

// NewWebRTCAPI creates a webrtc.API with production interceptors:
// NACK generator/responder, RTCP Reports, TWCC, GCC Pacer.
// Returns (api, estimatorChan) — estimatorChan delivers one BandwidthEstimator per PeerConnection.
func NewWebRTCAPI(cfg WebRTCAPIConfig) (*webrtc.API, <-chan cc.BandwidthEstimator, error) {
	if cfg.InitialBitrate <= 0 {
		cfg.InitialBitrate = 1_000_000
	}
	if cfg.MinBitrate <= 0 {
		cfg.MinBitrate = 100_000
	}
	if cfg.MaxBitrate <= 0 {
		cfg.MaxBitrate = 10_000_000
	}

	m := &webrtc.MediaEngine{}
	if err := m.RegisterDefaultCodecs(); err != nil {
		return nil, nil, err
	}

	i := &interceptor.Registry{}

	// 1) GCC + NoOp pacer (must be before RegisterDefaultInterceptors).
	// Real-time pacing lives in DirectPeer.publishAVSegment; leaky-bucket pacing
	// plus GCC dropping to MinBitrate during segment idle mis-queues RTP.
	congestionController, err := cc.NewInterceptor(func() (cc.BandwidthEstimator, error) {
		return gcc.NewSendSideBWE(
			gcc.SendSideBWEInitialBitrate(cfg.InitialBitrate),
			gcc.SendSideBWEMinBitrate(cfg.MinBitrate),
			gcc.SendSideBWEMaxBitrate(cfg.MaxBitrate),
			gcc.SendSideBWEPacer(gcc.NewNoOpPacer()),
		)
	})
	if err != nil {
		return nil, nil, err
	}

	estimatorChan := make(chan cc.BandwidthEstimator, 1)
	congestionController.OnNewPeerConnection(func(id string, estimator cc.BandwidthEstimator) {
		estimatorChan <- estimator
	})
	i.Add(congestionController)

	// 2) TWCC header extension (required by GCC, must be before RegisterDefaultInterceptors)
	if err := webrtc.ConfigureTWCCHeaderExtensionSender(m, i); err != nil {
		return nil, nil, err
	}

	// 3) Default interceptors: NACK generator/responder, RTCP Reports, Stats, TWCC sender
	if err := webrtc.RegisterDefaultInterceptors(m, i); err != nil {
		return nil, nil, err
	}

	api := webrtc.NewAPI(
		webrtc.WithMediaEngine(m),
		webrtc.WithInterceptorRegistry(i),
	)
	return api, estimatorChan, nil
}
